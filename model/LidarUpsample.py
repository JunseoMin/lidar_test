import torch
import torch.nn as nn

import torch.nn.functional as F

from addict import Dict
from functools import partial

from timm.models.layers import DropPath
import spconv.pytorch as spconv

from .serialization import encode
from collections import OrderedDict
from addict import Dict

import sys
import math
import torch_scatter

from einops import rearrange


# Awsome PTv3 codes (not modified) ---------------------------------
# Thanks to authors of PTv3!

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length()
        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(
                torch.max(self.grid_coord, dim=0).values, pad
            ).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat(
                [self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

class PointModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(
                            input.feat
                        )
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent



class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point

#-----------------------------------------------------

class MLP(nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout = 0.
                 ):
        super().__init__()

        self.l1 = nn.Linear(in_channels,hidden_channels)
        self.l2 = nn.Linear(hidden_channels,out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x



class CloseSerializedAttn(PointModule):
    def __init__(self,
                 channels,
                 patch_size,
                 n_heads,
                 qk_scale = None,
                 atten_bias = True,
                 atten_dropout = 0.,
                 proj_dropout = 0.,
                 order_idx = 0,
                 ):
        super().__init__()

        self.channels = channels
        self.n_heads = n_heads

        assert channels % n_heads == 0, "channel should be devided by the number of heads"

        self.order_idx = order_idx
        self.scale = qk_scale or channels//n_heads ** -0.5

        self.patch_size_max = patch_size
        self.patch_size = 0
        self.attn_drop = torch.nn.Dropout(atten_dropout)

        self.qkv = nn.Linear(channels, channels * 3, bias=atten_bias)
        
        self.projection = nn.Linear(channels,channels)  
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim = -1)


    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_idx}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            relative_positions = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
            point[rel_pos_key] = relative_positions
        return point[rel_pos_key]

    # @torch.no_grad()
    # def calculate_distance_weights(self, relative_positions):
    #     distances_squared = torch.sum(relative_positions**2, dim=-1)
    #     return torch.exp(-distances_squared / (2 * self.sigma**2))
    
    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                #print("!!!!")
                #print(self.patch_size)
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        """
        @torch.inference_mode()
        def offset2bincount(offset):
            return torch.diff(
                offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
            )
        """
        self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )
        
        H = self.n_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_idx][pad]
        #print("attndebug ===========")
        #print(point.serialized_order[self.order_idx])
        #print(order)
        #print("attndebug ===========")
        inverse = unpad[point.serialized_inverse[self.order_idx]]

        qkv = self.qkv(point.feat)[order]
        # #print(qkv)

        q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        # relative_positions = self.get_rel_pos(point, order)
        # distance_weights = self.calculate_distance_weights(relative_positions)
        # distance_weights = torch.log(distance_weights)
        # attn = attn + distance_weights

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(qkv.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, C)

        feat = feat[inverse]
        feat = self.projection(feat)
        feat = self.proj_dropout(feat)        

        point.feat = feat
        return point
        

class Block(PointModule):
    def __init__(self,
                 channels,
                 cpe_indice_key,
                 patch_size,
                 n_heads,
                 qkv_bias,
                 qk_scale,
                 attn_drop,
                 proj_drop,
                 order_index,
                 mlp_ratio,
                 drop_path,
                 norm_layer = None
                 ):
        super().__init__()
        self.channels = channels

        self.xcpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=5,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = CloseSerializedAttn(
            channels=channels,
            patch_size=patch_size,
            n_heads=n_heads,
            atten_bias=qkv_bias,
            qk_scale=qk_scale,
            atten_dropout=attn_drop,
            proj_dropout=proj_drop,
            order_idx=order_index
        )
        
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                dropout=proj_drop
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.xcpe(point)

        point.feat = shortcut + point.feat
        shortcut = point.feat

        point = self.norm1(point)
        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat

        shortcut = point.feat
        point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        #print(f"block output feat shape: {point.feat.shape}")
        return point


class FeatureExpending(PointModule):
    r"""
    Expand the featuremap to the upsample ratio with C2 channel and reshape
    """
    def __init__(self, in_feat_channels, c1 , c2, upsample_ratio):
        super().__init__()
        self.upsample_ratio = upsample_ratio
        
        self.conv1x1sets = nn.ModuleList()
        for i in range(upsample_ratio):
            conv_set = nn.Sequential(
                nn.Conv1d(in_feat_channels, c1, kernel_size=1),
                nn.BatchNorm1d(c1),
                nn.GELU(),
                nn.Conv1d(c1, c2, kernel_size=1),
                nn.BatchNorm1d(c2),
                nn.GELU()
            )
            self.conv1x1sets.append(conv_set)
        
    def forward(self, point):
        #print(f"expending input feat shape: {point.feat.shape}")
        # seperate the feature into different channels
        feat = rearrange(point.feat, 'n c -> 1 c n')
        
        concated_feat = []
        
        for convset in self.conv1x1sets:
            tmp_feat = convset(feat)
            concated_feat.append(tmp_feat)
        
        concated_feat = torch.cat(concated_feat, dim=1)
        
        #print(f"expending output feat shape: {concated_feat.shape}")
        point.feat = rearrange(concated_feat, '1 (c r) n -> (r n) c' , r = self.upsample_ratio)    #rearrange the feature channel  [N , rC] -> [rN , C]
        #print(f"expending output feat shape(rearranged): {point.feat.shape}")
        return point
    
class FC(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels = 3, dropout = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class Lidar4US(PointModule):
    def __init__(self, 
                 in_channels,
                 drop_path,
                 block_depth,
                 enc_channels,
                 enc_n_heads,
                 enc_patch_size,
                 qkv_bias, 
                 qk_scale, 
                 attn_drop, 
                 proj_drop, 
                 mlp_ratio, 
                 stride,
                 dec_depths,
                 dec_n_head,
                 dec_patch_size,
                 dec_channels,
                 train_decoder = True,
                 order = ("z", "z-trans", "hilbert", "hilbert-trans"),
                 upsample_ratio = 32,
                 out_channel=3,
                 fc_hidden=512,
                 exp_hidden=1024,
                 exp_out=256
                ):
        
        super().__init__()
        n_stage = len(block_depth)
        self.order = order
        self.encoder = PointSequential()
        self.train_decoder = train_decoder
        self.generate_encoder(drop_path, 
                              n_stage,
                              block_depth, 
                              enc_channels, 
                              enc_n_heads,
                              enc_patch_size, 
                              qkv_bias, 
                              qk_scale, 
                              attn_drop, 
                              proj_drop, 
                              mlp_ratio, 
                              stride
                              )

        assert len(enc_channels) == n_stage, "Encoder channels must match number of stages"
        assert len(stride) == n_stage - 1, "Stride must match number of stages - 1"
        assert all(ps > 0 for ps in enc_patch_size), "Patch sizes must be positive"
        assert all(ps > 0 for ps in dec_patch_size), "Patch sizes must be positive"

        self.decoder = PointSequential()
        self.generate_decoder(drop_path,
                              dec_depths,
                              enc_channels,
                              dec_channels,
                              n_stage,
                              dec_n_head,
                              dec_patch_size,
                              mlp_ratio,
                              qkv_bias,
                              qk_scale,
                              attn_drop,
                              proj_drop,
                              )

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            act_layer=nn.GELU,
        )
        
        self.expending = PointSequential()
        self.expending.add(FeatureExpending(
            in_feat_channels=dec_channels[0], # + in_channels
            c1 = exp_hidden,
            c2 = exp_out,
            upsample_ratio=upsample_ratio
        ), name="expending")
        
        self.fc = PointSequential(
            FC(
                in_channels=exp_out,
                hidden_channels=fc_hidden,
                out_channels=out_channel,
                dropout=0.0
            )
        )
        
    def generate_encoder(self, 
                         drop_path, 
                         n_stage,
                         block_depth, 
                         enc_channels, 
                         enc_n_heads:tuple,
                         enc_patch_size, 
                         qkv_bias, 
                         qk_scale, 
                         attn_drop , 
                         proj_drop, 
                         mlp_ratio, 
                         stride:tuple ):
        
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(block_depth))]
        for stage in range(n_stage):
            _enc_drop_path = enc_drop_path[sum(block_depth[:stage]):sum(block_depth[:stage + 1])]
            enc = PointSequential()

            if stage == 0:
                for i in range(block_depth[stage]):
                    enc.add(
                        Block(
                            channels=enc_channels[stage],
                            n_heads=enc_n_heads[stage],
                            patch_size=enc_patch_size[stage],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=_enc_drop_path[i],
                            norm_layer=partial(
                                            PDNorm,
                                            norm_layer=partial(nn.LayerNorm, elementwise_affine=True),
                                            conditions=("ScanNet", "S3DIS", "Structured3D"),
                                            decouple=True,
                                            adaptive=False,
                                        ),
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{stage}",
                        ),
                        name=f"block{i}",
                    )
         
            if stage > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[stage - 1],
                        out_channels=enc_channels[stage],
                        stride=stride[stage - 1],
                        norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                        act_layer=nn.GELU,
                    ),
                    name="down",
                )

                for i in range(block_depth[stage]):
                    enc.add(
                        Block(
                            channels=enc_channels[stage],
                            n_heads=enc_n_heads[stage],
                            patch_size=enc_patch_size[stage],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=_enc_drop_path[i],
                            norm_layer= nn.LayerNorm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{stage}"
                        ),
                        name=f"block{i}",
                    )

                if len(enc) != 0:
                    self.encoder.add(module=enc, name=f"enc{stage}")


    def generate_decoder(self,
                         drop_path,
                         dec_depths,
                         enc_channels,
                         dec_channels,
                         n_stage,
                         dec_n_head:tuple,
                         dec_patch_size:tuple,
                         mlp_ratio,
                         qkv_bias,
                         qk_scale,
                         attn_drop,
                         proj_drop,
                         ):
        dec_drop_path = [ x.item() for x in torch.linspace(0, drop_path, sum(dec_depths)) ]
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        for stage in reversed(range(0,n_stage - 1)):
            _dec_drop_path = dec_drop_path[sum(dec_depths[:stage]) : sum(dec_depths[:stage + 1])]

            _dec_drop_path.reverse()
            dec = PointSequential()

            dec.add(
                SerializedUnpooling(
                        in_channels=dec_channels[stage + 1],
                        skip_channels=enc_channels[stage],
                        out_channels=dec_channels[stage],
                        norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                        act_layer=nn.GELU,
                        ),
                        "up"
            )
            for i in range(dec_depths[stage]):
                    dec.add(
                        Block(
                            channels=dec_channels[stage],
                            n_heads=dec_n_head[stage],
                            patch_size=dec_patch_size[stage],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=_dec_drop_path[i],
                            norm_layer=nn.LayerNorm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{stage}",
                        ),
                        name=f"block{i}",
                    )
            self.decoder.add(module=dec, name=f"dec{stage}")

    def forward(self, data_dict):
        point = Point(data_dict)
        # point_raw_feat = point.feat

        point.serialization(self.order, shuffle_orders=True)
        point.sparsify()
        
        #print(f"embedding input feat shape: {point.feat.shape}")
        point = self.embedding(point)
        #print(f"embedding output feat shape: {point.feat.shape}")
        descriptor = self.encoder(point)

        if self.train_decoder:
            point = self.decoder(descriptor)
            # point.feat = torch.cat((point.feat, point_raw_feat), dim=1)  #skip connection
            
            point = self.expending(point)
            point = self.fc(point)
            #print(f"fc output feat shape: {point.feat.shape}")
            return point

        return descriptor