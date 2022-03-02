from curses import window
from re import X
import sys,os

from cv2 import norm
from matplotlib import use
from matplotlib.cbook import normalize_kwargs
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
    B_, w_size, w_size, C = windows.shape
    assert w_size == window_size, "the window size doesn't match!!"
    h_num = H // window_size
    w_num = W // window_size
    num_windows = h_num * w_num 
    B = B_ // num_windows
    windows = windows.view(B, h_num, w_num, window_size, window_size, C).contigous().permute(0, 1, 3, 2, 4, 5)
    x = windows.view(B, H, W, C)
    return x




    
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

        if shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros(1, H, W, 1)

            h_slices = [slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None)]
            w_slices = [slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None)]
            
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[1, h, w, 1] = cnt
                    cnt += 1
            mask_window = window_partition(img_mask, window_size)
            mask_window = mask_window.view(-1, window_size, window_size)
            attn_mask = mask_window.unsqueeze(1) - mask_window.unsqueeze(2)
            attn_mask = mask_window.masked_fill(attn_mask!=0, -100.0).masked_fill(attn_mask==0, 0.)
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        '''
        x's shape: [B, N, C]
        '''
        H, W = self.input_resolution
        B, N, C = x.shape
        assert N == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shift_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                                dims=(1, 2))
        else:
            shift_x = x

        x = window_partition(shift_x, self.window_size)
        x = x.view(x.shape[0], -1, C)
        
        attn = self.attn(x, mask=self.attn_mask).view(x.shape[0], self.window_size, self.window_size, C)

        attn = window_reverse(attn, self.window_size, H, W)
        
        if self.shift_size > 0:
            shift_x = torch.roll(attn, shifts=(self.shift_size, self.shift_size),
                                dims=(1, 2))
        else:
            shift_x = attn
        
        out = self.drop_path(shift_x.view(B, -1, C)) + shortcut

        shortcut = out
        out = self.drop_path(self.mlp(self.norm2(out))) + shortcut

        return out

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)


    
    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H, W = self.input_resolution
        assert L == H * W, "the shape of H, W and L doesn't match" 
        assert self.dim == C, "the dim and the Channels don't match"

        x = x.view(B, H, W, C)
        X0 = x[:, 0::2, 0::2, :]        
        X1 = x[:, 1::2, 0::2, :]
        X2 = x[:, 1::2, 1::2, :]
        X3 = x[:, 0::2, 1::2, :]
        x = torch.cat([X0, X1, X2, X3], dim=-1)

        x = x.view(B, -1, 4*C)
        x = self.norm(x)

        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops



 
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, 
            mlp_ratio, qkv_bias, qk_scale, drop=0.0, attn_drop=0.0, drop_path=0.0,
            norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        assert depth % 2 == 0, "the depth of the layer must be zero."

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        
        self.Swin_T = nn.ModuleList([SwinTransformerBlock(dim, input_resolution, num_heads, window_size, 
                                        0 if i % 2 == 0 else window_size //2, mlp_ratio, qkv_bias, qk_scale, 
                                        drop, attn_drop=attn_drop, drop_path=drop_path) 
                                                for i in depth])
        
        self.use_checkpoint = use_checkpoint

    def forward(self, x):

        for block in self.Swin_T:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        if self.downsample:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


 
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)

        self.norm = norm_layer

    def forward(self, x):
        '''
        x's shape: [B, C, H, W]
        '''

        x = self.proj(x).flatten(2).transpose(1, 2)

        if self.norm:
            x = self.norm(x)
        
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

 
class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch implemention of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, 
                embed_dim=96, depths=[2, 2, 6, 2], nums_head=[3, 6, 12, 24], 
                window_size=7, mlp_ratio=4, qkv_bias=True, qk_scale=None, 
                drop_rate=0, attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, 
                ape=False, patch_norm=True, use_checkpoint=False):
        super().__init__()

        # split the image into the non-overlap patches
        self.patch_partition = PatchEmbed(img_size, patch_size, in_chans, 
                                    embed_dim, norm_layer)
        
        self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i, (depth, head) in enumerate(zip(depths, nums_head)):
            layer = BasicLayer(embed_dim * (2 ** 2), (img_size[0]//(4 * 2**i), img_size[1] // (4 * 2**i)),
                                 depth, head, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate, 
                                 attn_drop_rate, drop_path_rate, norm_layer, downsample=PatchMerging if i < len(depths)-1 else None, use_chekpoint=use_checkpoint)
            self.layers.append(layer)
        
        self.norm = norm_layer(embed_dim * 2 ** (len(depths) - 1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 2 ** (len(depths) - 1), num_classes)
    
 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_partition(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


if __name__ == "__main__":
    

    















    # import numpy as np 

    # # test the window_patition:
    # x = torch.randn(32, 21, 21, 3)
    # window_size1 = 7
    # window_size2 = 8
    # out1 = window_partition(x, window_size=window_size1)
    # # out2 = window_partition(x, window_size2)
    # # out3 = window_partition_(x, window_size1)
    # print("out1 size:", out1.shape)
    # # print("out2 size:", out2.shape)
    # # print("out3 size:", out3.shape)
    # # print("out1 == out3?:", np.any(np.array(out1 == out3)))

    # # @ operator
    # a = x @ torch.randn(3, 4)
    # print(a.shape)

    # # test the position code of transformer
    # window_size = [7, 7]
    # coords_h = torch.arange(window_size[0])
    # coords_w = torch.arange(window_size[1])
    # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    # print("the meshgrid:", torch.meshgrid([coords_h, coords_w]))
    # print("the stack:", coords)
    # print("the shape of stack :", coords.shape)
    # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # print("the shape of coords_flatten:", coords_flatten.shape)
    # print("the matrix of coords_flatten:", coords_flatten)
    # relative_coords = coords_flatten[:, :, None] - \
    #     coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    # print("relative_coords:", relative_coords.shape)
    # print("the shape of coords_flatten[:, :, None]", coords_flatten[:, :, None].shape)
    # print("the shape of coords_flatten[:, None, :]", coords_flatten[:, None, :].shape)
    # relative_coords = relative_coords.permute(
    #     1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # relative_coords[:, :, 0] += window_size[0] - \
    #     1  # shift to start from 0
    # relative_coords[:, :, 1] += window_size[1] - 1
    # relative_coords[:, :, 0] *= 2 * window_size[1] - 1
    # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    # print("the position_index 's shape", relative_position_index)

    # # # define a parameter table of relative position bias
    # # self.relative_position_bias_table = nn.Parameter(
    # #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

    # # # get pair-wise relative position index for each token inside the window
    # # coords_h = torch.arange(self.window_size[0])
    # # coords_w = torch.arange(self.window_size[1])
    # # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    # # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    # # relative_coords = coords_flatten[:, :, None] - \
    # #     coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    # # relative_coords = relative_coords.permute(
    # #     1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    # # relative_coords[:, :, 0] += self.window_size[0] - \
    # #     1  # shift to start from 0
    # # relative_coords[:, :, 1] += self.window_size[1] - 1
    # # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
    # # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    # # self.register_buffer("relative_position_index",
    # #                      relative_position_index)

    # # test the torch.roll
    # # cyclic shift
    # shift_size = 4
    # x = torch.rand(32, 224, 224, 3)
    # if shift_size > 0:
    #     shifted_x = torch.roll(
    #     x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    # else:
    #     shifted_x = x



