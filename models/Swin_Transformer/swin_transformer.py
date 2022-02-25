import sys,os

from cv2 import norm
sys.path.append(os.getcwd())
from turtle import forward
import torch
from torch import nn
from torch.utils import checkpoint
import torch.nn.functional as F
import math

from utils.weight_init import trunc_normal_
from utils.dot_product import scaled_dot_product
from utils.tools import DropPath

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num, acti_layer=nn.GELU, dropout=0.0):
        super().__init__()
        self.layer_num = layer_num
        hiddens = [hidden_dim] * (layer_num-1)
        self.layers = [nn.Linear(in_, out_) for in_, out_ in zip([in_dim] + hiddens, hiddens + [out_dim])]
        self.drop = nn.Dropout(dropout)
        self.activ = acti_layer
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers, 1):
           out = self.drop(self.activ(layer(x))) if i < self.layer_num else self.drop(layer(x))
        return out

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)

    Attention!!: This pic must be padded before! hahaha.
    """
    B, H, W, C = x.shape
    raw = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = raw.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)    
    # 官网参考代码给出了.contiguous().view来转换，如果直接在permute后使用view会报错，
    # 但是用reshape就不会， 至于内部具体的原理，这里不多介绍，可以参考博客：https://zhuanlan.zhihu.com/p/64551412。
    return windows



# def window_partition_(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows


       


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    pass


    
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        head_dim = dim // num_heads
        self.scale = qk_scale or math.sqrt(head_dim)
        self.num_heads = num_heads
        self.window_size = window_size

        # position table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size[0] - 1, 2*window_size[1] -1, num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        
        B_, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B_, N,  self.num_heads, C // self.num_heads, 3).permute(0, 2, 1, 3, 4).chunck(3, dim=-1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        bias = relative_position_bias.unsqueeze(0)

        attn = scaled_dot_product(q, k, v, bias=bias)
        attn = self.attn_drop(attn)

        attn = attn.permute(0, 2, 1, 3)
        attn = attn.reshape(B_, N, -1)

        out = self.proj(attn)
        out = self.proj_drop(out)

        return out

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        
        
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size, shift_size,
         mlp_ratio, qkv_bias, qk_scale, drop=0.0, attn_drop=0.0, 
         drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.shift_size = shift_size
        self.window_size = window_size

        if self.window_size >= min(self.input_resolution):
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
        assert 0<=self.shift_size<=self.window_size, "shift size must be smaller than the window size."

        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        hidden_mlp = dim * mlp_ratio
        self.mlp = MLP(dim, hidden_mlp, out_dim=dim, layer_num=3, acti_layer=act_layer, dropout=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        








 




















if __name__ == "__main__":

    import numpy as np 

    # test the window_patition:
    x = torch.randn(32, 21, 21, 3)
    window_size1 = 7
    window_size2 = 8
    out1 = window_partition(x, window_size=window_size1)
    # out2 = window_partition(x, window_size2)
    # out3 = window_partition_(x, window_size1)
    print("out1 size:", out1.shape)
    # print("out2 size:", out2.shape)
    # print("out3 size:", out3.shape)
    # print("out1 == out3?:", np.any(np.array(out1 == out3)))

    # @ operator
    a = x @ torch.randn(3, 4)
    print(a.shape)

    # test the position code of transformer
    window_size = [7, 7]
    coords_h = torch.arange(window_size[0])
    coords_w = torch.arange(window_size[1])
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    print("the meshgrid:", torch.meshgrid([coords_h, coords_w]))
    print("the stack:", coords)
    print("the shape of stack :", coords.shape)
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    print("the shape of coords_flatten:", coords_flatten.shape)
    print("the matrix of coords_flatten:", coords_flatten)
    relative_coords = coords_flatten[:, :, None] - \
        coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    print("relative_coords:", relative_coords.shape)
    print("the shape of coords_flatten[:, :, None]", coords_flatten[:, :, None].shape)
    print("the shape of coords_flatten[:, None, :]", coords_flatten[:, None, :].shape)
    relative_coords = relative_coords.permute(
        1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - \
        1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    print("the position_index 's shape", relative_position_index)




    # # define a parameter table of relative position bias
    # self.relative_position_bias_table = nn.Parameter(
    #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

    # # get pair-wise relative position index for each token inside the window
    # coords_h = torch.arange(self.window_size[0])
    # coords_w = torch.arange(self.window_size[1])
    # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # relative_coords = coords_flatten[:, :, None] - \
    #     coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    # relative_coords = relative_coords.permute(
    #     1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # relative_coords[:, :, 0] += self.window_size[0] - \
    #     1  # shift to start from 0
    # relative_coords[:, :, 1] += self.window_size[1] - 1
    # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    # self.register_buffer("relative_position_index",
    #                      relative_position_index)
