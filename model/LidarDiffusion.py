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
import copy

import numpy as np

import flash_attn

# Awsome PTv3 codes (not modified) ---------------------------------
# Thanks to authors of PTv3!
"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
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
            depth = min(int(self.grid_coord.max()).bit_length(), 16)
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
        # print("pooling!")
        # print(point.feat.shape)
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
        traceable=True,  # record parent and cluster
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

class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        n_heads,
        patch_size,
        attn_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
    ):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.scale = qk_scale or (channels // n_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=attn_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, n_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

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
        
        H = self.n_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        
        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.half().reshape(-1, 3, H, C // H),
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, C)
        feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
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
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            n_heads=n_heads,
            attn_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index
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

class PointTransformerEncoder(PointModule):
    def __init__(self, 
                 drop_path, 
                 n_stage,
                 enc_block_depth, 
                 enc_channels, 
                 enc_n_heads:tuple,
                 enc_patch_size, 
                 qkv_bias, 
                 qk_scale, 
                 attn_drop , 
                 proj_drop, 
                 mlp_ratio, 
                 stride:tuple,
                 order,
                 in_channels,
                 condition_out_channel,
                 condition_hidden_channel
                 ):
        super().__init__()

        self.order = order
        self.encoder = PointSequential()

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
            act_layer=nn.GELU,
        )
    
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_block_depth))]
        for stage in range(n_stage):
            _enc_drop_path = enc_drop_path[sum(enc_block_depth[:stage]):sum(enc_block_depth[:stage + 1])]
            enc = PointSequential()

            if stage == 0:
                for i in range(enc_block_depth[stage]):
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

                for i in range(enc_block_depth[stage]):
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

        self.fc = PointSequential(FC(
            enc_channels[-1],
            hidden_channels=condition_hidden_channel,
            out_channels=condition_out_channel
        ))

    def forward(self, data_dict):
        point = Point(data_dict)

        point.serialization(self.order, shuffle_orders=True)
        point.sparsify()
        
        point = self.embedding(point)
        condition = self.encoder(point)
        return self.fc(condition).feat

class AutoEncoder(PointModule):
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
                 order = ("z", "z-trans", "hilbert", "hilbert-trans"),
                 out_channel=3,
                 fc_hidden=32,
                ):
        
        super().__init__()
        n_stage = len(block_depth)
        self.order = order
        self.encoder = PointSequential()
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

        self.decoder = self.generate_decoder(drop_path,
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
        
        self.fc = PointSequential(
            FC(
                in_channels=dec_channels[0],
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
        decoder = PointSequential()
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
            decoder.add(module=dec, name=f"dec{stage}")
        
        return decoder

    def forward(self, data_dict):
        point = Point(data_dict)

        point.serialization(self.order, shuffle_orders=True)
        point.sparsify()
        
        point = self.embedding(point)
        point = self.encoder(point)

        point = self.decoder(point)
        point = self.fc(point)

        return point.feat
    
class LiDARDiffusion(PointModule):
    #code referenced from: PTv3 & https://github.com/luost26/diffusion-point-cloud
    def __init__(self, 
                      condition_drop_path = 0.3, 
                      condition_enc_block_depth = (2, 2, 2), 
                      condition_enc_channels = (32, 64, 128), 
                      condition_enc_n_heads = (2, 4, 8),
                      condition_enc_patch_size = (512, 512, 512), 
                      condition_qkv_bias = True, 
                      condition_qk_scale = None, 
                      condition_attn_drop  = 0.1, 
                      condition_proj_drop = 0.1, 
                      condition_mlp_ratio = 4, 
                      condition_stride = (2, 2, 2), 
                      condition_in_channels = 4,
                      condition_out_channel = 128,
                      condition_hidden_channel = 32,
                      drop_path = 0.3,
                      enc_block_depth = (2,2,2,4,2),
                      enc_channels = (32, 64, 128, 256, 512),
                      enc_n_heads = (2, 4, 8, 16, 32),
                      enc_patch_size = (1024, 1024, 1024, 1024, 1024), 
                      qkv_bias = True, 
                      qk_scale = None, 
                      attn_drop = 0.1, 
                      proj_drop = 0.1, 
                      mlp_ratio = 4, 
                      stride = (2, 2, 2, 2),
                      order=("z", "z-trans", "hilbert", "hilbert-trans"),
                      dec_depths = (2, 2, 2, 2),
                      dec_channels = (32, 64, 128, 256),
                      dec_n_head = (2, 4, 8, 16),
                      dec_patch_size = (1024, 1024, 1024, 1024),
                      time_out_ch = 3,
                      gt_channels = 3,
                      num_steps = 1000,
                      beta_1 = 10e-5,
                      beta_T = 10e-2,
                      device = 'cuda'
                      ):
        super().__init__()
        self.device = device
        condition_n_stage = len(condition_enc_block_depth)
        n_stage = len(enc_block_depth)

        self.condition_encoder = PointTransformerEncoder( 
                                                condition_drop_path, 
                                                condition_n_stage,
                                                condition_enc_block_depth, 
                                                condition_enc_channels, 
                                                condition_enc_n_heads,
                                                condition_enc_patch_size, 
                                                condition_qkv_bias, 
                                                condition_qk_scale, 
                                                condition_attn_drop , 
                                                condition_proj_drop, 
                                                condition_mlp_ratio, 
                                                condition_stride,
                                                order,
                                                condition_in_channels,
                                                condition_out_channel,
                                                condition_hidden_channel
                                                )

        in_channels = gt_channels + condition_out_channel + time_out_ch
        
        self.model = AutoEncoder(in_channels= in_channels,
                                 drop_path=drop_path,
                                 block_depth=enc_block_depth,
                                 enc_channels=enc_channels,
                                 enc_n_heads=enc_n_heads,
                                 enc_patch_size=enc_patch_size,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop,
                                 proj_drop=proj_drop,
                                 mlp_ratio=mlp_ratio,
                                 stride=stride,
                                 dec_depths=dec_depths,
                                 dec_n_head=dec_n_head,
                                 dec_patch_size=dec_patch_size,
                                 dec_channels=dec_channels,
                                 order=order,
                                 out_channel=3,
                                 fc_hidden=64
                                 )
        self.time_emb_ch = time_out_ch
        self.var_sched = VarianceSchedule(num_steps=num_steps,   # max T
                                          beta_1=beta_1,
                                          beta_T=beta_T
                                          )
           
    def get_loss(self, static_objects, lidar_16, device, t=None):
        r'''
        args: 
            static object: for train, GT 
            time_embedding: sinusoidal positional embedded time_embedding
            lidar_16: 16ch lidar raw 
        '''
        latent_z = self.condition_encoder(lidar_16)
        L_dim , _ = latent_z.shape
        N, point_dim = static_objects.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(1)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(static_objects)  # (N, 3)
        x = self.make_dictionary(c0 * static_objects + c1 * e_rand, latent_z, beta, device=device)    #forward process
        e_theta = self.model(x)
        
        
        if N < L_dim:
            final_dim = max(point_dim,L_dim)
            assert L_dim > point_dim, "What?!?!"
            
            e_rand_pad_rows = final_dim - e_rand.shape[0]
            e_rand_zero_padding = torch.zeros((e_rand_pad_rows, 3), device=device, dtype=e_rand.dtype)
            e_rand_padded = torch.cat([e_rand, e_rand_zero_padding], dim=0)
        
            loss = F.mse_loss(e_theta.view(-1, final_dim), e_rand_padded.view(-1, final_dim), reduction='mean')
        else:
            loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')

        return loss
    
    def sample(self, num_points, lidar_16, device, t=None, flexibility = 0., is_validation = True):
        latent_z = self.condition_encoder(lidar_16)
        x_T = torch.randn([1, num_points, 3]).to(device)
        traj = {self.var_sched.num_steps: x_T}

        for t in range(self.var_sched.num_steps, 0, -1):
            condition = latent_z.clone().detach()
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            beta = self.var_sched.betas[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            x = self.make_dictionary(x_t, condition, beta, device=device)

            e_theta = self.model(x)
            # print(e_theta)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
        

        return traj[0] if is_validation else traj
    
    def make_dictionary(self, noised_input, latent_z, beta, device, grid_size=0.05, segments=1):
        """
        Create a dictionary containing coordinates, features, batch indices, and grid size.

        Args:
            noised_input (Tensor): Input tensor with shape (B, N, C), where B is the batch size, N is the number of points, and C is the feature dimension.
            latent_z (Tensor): Latent tensor with 'feat' attribute containing additional features.
            t (Tensor): Timesteps tensor used for sinusoidal embeddings.
            grid_size (float): Grid size for point cloud processing.
            segments (int): Number of segments for batch creation.

        Returns:
            dict: A dictionary with 'coord', 'feat', 'batch', and 'grid_size'.
        """
        # Extract latent features and generate sinusoidal time embeddings
        B, N, C = noised_input.shape
        beta = beta.view(1, 1).to(device)
        # Compute shapes
        
        t_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=1)  # (B, 1, 3)
        D, L = latent_z.shape  # latent: (B, D)

        # Determine max padding size across tensors
        max_dim = max(N, D)

        # Pad latent features
        latent_pad_rows = max_dim - latent_z.shape[0]
        latent_zero_padding = torch.zeros((latent_pad_rows, L), device=device, dtype=latent_z.dtype)
        latent_padded = torch.cat([latent_z, latent_zero_padding], dim=0)  # (B*max_dim, D)

        # Rearrange coordinates
        coord = rearrange(noised_input, "b n c -> (b n) c")  # e.g. (B*N, 3)
        coord_pad_rows = max_dim - coord.shape[0]
        coord_zero_padding = torch.zeros((coord_pad_rows, 3), device=device, dtype=noised_input.dtype)
        coord_padded = torch.cat([coord, coord_zero_padding], dim=0)  # (B*max_dim, C)

        # Pad time embeddings
        t_emb_pad_rows = max_dim - t_emb.shape[0]
        t_emb_zero_padding = torch.zeros((t_emb_pad_rows, 3), device=device, dtype=t_emb.dtype)
        t_emb_padded = torch.cat([t_emb, t_emb_zero_padding], dim=0)  # (B*max_dim, T)

        # Concatenate features for diffusion
        feat = torch.cat([coord_padded, latent_padded, t_emb_padded], dim=1)

        # Create batch tensor
        batch_tensor = torch.full((max_dim,), 0, dtype=torch.int64).to(device)

        # Create and return the dictionary
        return {
            "coord": coord_padded,
            "feat": feat,
            "batch": batch_tensor,
            "grid_size": torch.tensor(grid_size, device=noised_input.device)
        }

class VarianceSchedule(nn.Module):
    #code from :https://github.com/luost26/diffusion-point-cloud/blob/main/models/diffusion.py
    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas