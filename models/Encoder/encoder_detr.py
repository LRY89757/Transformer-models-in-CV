import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import sys,os
sys.path.append(os.getcwd())
from utils.tools import _getactiv


class EncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads:int, mlp_dim=256,
                 activ = "relu", dropout=0.0, norm_pre=True):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(dropout),
            _getactiv(activ),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout()
        )

        self.norm_pre = norm_pre

    def pos_emb(self, src, pos):
        return src if pos is None else src + pos
    
    def forward_pre(self, tgt, pos):
        src = self.norm1(tgt)
        q = k = self.pos_emb(src, pos)
        v = src

        src = self.attn(q, k, v)[0]   # return the tuple, so we should end with "[0]" to get the src
        src = self.drop(src)

        src1 = src + tgt

        src2 = self.norm2(src1)
        src2 = self.mlp(src2)

        out = src2 + src1
        return out


    def forward_post(self, tgt, pos):
        q = k = self.pos_emb(tgt, pos)
        v = tgt 

        src = self.attn(q, k, v)[0]
        src1 = self.drop(src)
        src1 = src1 + src
        src2 = self.norm1(src1)

        src3 = self.mlp(src2)
        out = self.norm2(src2 + src3)
        return out

    def forward(self, tgt, pos:Optional[Tensor] = None):
        '''
        tgt's shape: [T, B, D]
        pos's shape: [T, B, D]
        '''
        # print("hello")
        if self.norm_pre:
            return self.forward_pre(tgt, pos)
        else:
            return self.forward_post(tgt, pos)






class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, embed_dim, num_heads:int, mlp_dim,
                 activ = "relu", dropout=0.1, norm_pre=True):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(embed_dim, num_heads, mlp_dim, activ, dropout, norm_pre)
                                         for i in range(num_layers)])        
    
    def forward(self, tgt, pos:Optional[Tensor] = None):
        out = tgt
        for layer in self.encoder:
            out = layer(out, pos)
        return out


if __name__ == "__main__":
    x = torch.randn(50, 64, 32) # [T, B, D]
    pos = torch.rand(50, 64, 32)
    num_layers, embed_dim, num_heads, mlp_dim = 8, 32, 8, 256
    encoder = TransformerEncoder(num_layers, embed_dim, num_heads, mlp_dim, activ = "relu", dropout=0.0, norm_pre=True)
    print(encoder(x, pos).shape)


