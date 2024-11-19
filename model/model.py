r"""
TODO: serialization encoder, cpe, grid pool , embedding, model constructor - 11/20
TODO(2): decoder - 11/21
test&eval ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spconv.pytorch as spconv

from addict import Dict

import flash_attn

#-----------------------serialization codes-------------------------
def zcurve():
    pass

def hilbert():
    pass

serializemethods = [zcurve,hilbert]

def serialize(lidar_points:torch.Tensor):
    serialized = []

    for method in serializemethods:
        serialized.append(method(lidar_points))

    return serialized

def encode():
    pass
#Type Class------------------------------------------------------

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

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)
        
    def serialization(self, order , depth=None, shuffle_orders = False):
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys(): #  leverage grid sampling
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        if depth is None:
            depth = int(self.grid_coord.max()).bit_length()
        
        self["serialized_depth"] = depth

        code = [encode(self.grid_coord, self.batch, depth, order= _order) for _order in order]

        code = torch.stack(code)
        order = torch.argsort(code)

        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1),
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
    

#Modules---------------------------------------------------------
class Attention(nn.Module): #done (11/19)
    r"""
    serialized attention - PTv3
    KNN-based patch grouping costs too much then Serialized Attention.

    methods:
    revisit and adopt the efficiant window and dot-product attention

    1. groups points into non-overlapping patches
    2. perform attention attention within each individual patch

    ** patch mergeing & patch grouping
    
    """
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
        self.patch_size = patch_size
        self.n_heads = n_heads

        assert channels % n_heads == 0, "channel should be devided by the number of heads"

        self.order_idx = order_idx
        self.scale = qk_scale or channels//n_heads ** -0.5

        self.atten_dropout = atten_dropout

        self.qkv = nn.Linear(channels, channels * 3, bias=atten_bias)
        
        # ffn
        self.projection = nn.Linear(channels,channels)  
        self.proj_dropout = F.dropout(proj_dropout)
        self.softmax = nn.Softmax(dim = -1)
        
    @torch.no_grad()
    def get_relative_pos(self, point:Point, order):
        k = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_idx}"

        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1,k,3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)

        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point:Point):
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
            _offset = F.pad(offset, (1, 0))
            _offset_pad = F.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
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
            point[cu_seqlens_key] = F.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point:Point):
        h = self.n_heads
        k = self.patch_size
        c = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_idx][pad]
        inverse = unpad[point.serialized_inverse[self.order_idx]]

        qkv = self.qkv(point.feat)[order]
        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, h, c // h),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, c)
        feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        feat = self.projection(feat)
        feat = self.proj_dropout(feat)

        point.feat = feat
        return point


class Mlp(nn.Module):
    r"""
    simple MLP layer
    """
    def __init__(self,
                 in_channel,
                 hidden_channels = None,
                 out_channels = None,
                 act_layer = nn.GELU,
                 dropout = 0.0
                 ):
        super().__init__()

        hidden_channels = hidden_channels or in_channel
        out_channels = out_channels or in_channel

        self.fc1 = nn.Linear(in_channel,hidden_channels)
        self.fc2 = nn.Linear(hidden_channels,out_channels)
        self.activation = act_layer
        self.dropout = nn.Dropout(dropout)


    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    
class xCPE():
    pass

class GridPool():
    pass

#layers---------------------------------------------------------
class EncoderLayer(nn.Module):
    def __init__(self,
                 n_encoders = 4,
                 n_blocks = [2,2,6,2]
                 ):
        super().__init__()

        assert n_encoders == len(n_blocks), "ERROR: Check the number of blocks"
        self.encoders = nn.ModuleList([Encoder(n_blocks=n_blocks[idx]) for idx in range(n_encoders)])   # n_blockx[idx] ==> idx encoder's number of block

    def forward(self,x):
        outs = []   # for U-Net style architecture

        for encoder in self.encoders:
            x = encoder(x)
            outs.append(x)
        
        return outs[::-1]   # for further iteration (first element is the last encoder output)

class Encoder(nn.Module):
    def __init__(self,
                 n_blocks:int,
                ):
        super().__init__()
        
        self.gridpool = GridPool()
        self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)]) 
        
    def forward(self,x):
        x = self.grid_pool(x)
        x = self.order_shuffle(x)

        for block in self.blocks:
            x = block(x)

        return x
        

class Block(nn.Module):
    def __init__(self,
                  channel, 
                  hidden_rate,
                  proj_drop
                  ):
        super().__init__()
        self.atten = Attention(channel,patch_size,n_heads,qk_scale,atten_bias,atten_dropout,proj_dropout,order_idx)
        self.cpe = xCPE()
        self.ln1 = nn.LayerNorm(channel)
        self.ln2 = nn.LayerNorm(channel)
        self.mlp = Mlp(in_channel=channel, hidden_channels=hidden_rate, out_channels=channel, dropout=proj_drop)

    def forward(self,x):
        residual = x
        x = self.cpe(x)
        x += residual
        residual = x

        x = self.ln1(x) # pre normalization: improved computation cost and spatial complexity(swin-t)
        x = self.atten(x)
        x += residual
        residual = x

        x = self.ln2(x) # pre norm
        x = self.mlp(x)
        x += residual

        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, 
                 in_channel,
                 embed_channel,
                 ln = None,
                 activation = None
                 ):
        super().__init__()


    def forward(self):
        

        pass

class DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward():
        pass

#------------------------------------------------------------------

#main model -------------------------------------------------------
class LidarUS(nn.Module):
    def __init__(self, 
                 n_blocks,
                 n_encoders
                 ):
        super().__init__()
        self.final = False

        self.encoderlayer = EncoderLayer(n_blocks=n_blocks, n_encoders= n_encoders)
        self.embeding = EmbeddingLayer()


    def forward(self,lidar_points):
        r"""
        
        """
        serailized = serialize(lidar_points)
        embedded = self.embeding(serailized)
        outs = self.encoderlayer(embedded)  #outs: each output of encoder layer (for U-Net shape)
        



        return point