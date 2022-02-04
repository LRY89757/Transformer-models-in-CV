import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import sys,os
sys.path.append(os.getcwd())
from utils.tools import _getactiv


class DecoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, activ="relu", dropout=0.1, mlp_dim=256, norm_pre=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.multi = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            _getactiv(activ),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)        

        self.norm_pre = norm_pre
    
    def pos_emb(self, src, pos):
        return src if pos is None else src + pos


    def forward_pre(self, query_obj, memory, 
                query_pos:Optional[Tensor]=None, pos:Optional[Tensor]=None):
        '''
        query_obj means the input of decoder.
        memory means the output of the encoder.
        query_pos means the position of query_obj.
        pos means the position of memory. 
        '''
        query_ = self.norm1(query_obj)
        q = k = self.pos_emb(query_, query_pos)
        v = query_
        att = self.attn(q, k, v)[0]
        src1 = query_obj + self.drop1(att)
        
        src2 = self.norm2(src1)
        q = self.pos_emb(src2, query_pos)
        k = self.pos_emb(memory, pos)
        v = memory
        attn = self.multi(q, k, v)[0]
        src3 = src1 + self.drop2(attn)

        src4 = self.norm3(src3)
        out = self.mlp(src4) + src3

        return out



    def forward_post(self, query_obj, memory,
                 query_pos:Optional[Tensor]=None, pos:Optional[Tensor]=None):
        assert query_obj is not None
        q = k = self.pos_emb(query_obj, query_pos)
        v = query_obj
        att = self.attn(q, k, v)[0]
        src1 = self.norm1(query_obj + self.drop1(att))

        q = self.pos_emb(src1, query_pos)
        k = self.pos_emb(memory, pos)
        v = memory
        attn = self.multi(q, k, v)[0]
        src2 = self.norm2(src1 + self.drop2(attn))

        src3 = self.norm3(self.mlp(src2) + src2)
        return src3

    def forward(self, query_obj, memory,
                 query_pos:Optional[Tensor]=None, pos:Optional[Tensor]=None):
        if self.norm_pre:
            return self.forward_pre(query_obj, memory, query_pos=query_pos, pos=pos)
        else:
            return self.forward_post(query_obj, memory, query_pos=query_pos, pos=pos)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, 
            num_heads=8, activ="relu", dropout=0.1, mlp_dim=512, norm_pre=True):
        super().__init__()
        self.decoder = nn.ModuleList([DecoderLayer(embed_dim, num_heads, activ, dropout, mlp_dim, norm_pre) 
                                                    for i in range(num_layers)])

    def forward(self, query_obj, memory,
                 query_pos:Optional[Tensor]=None, pos:Optional[Tensor]=None):
        out = query_obj
        for layer in self.decoder:
            out = layer(out, memory, query_pos=query_pos, pos=pos)

        return out.unsqueeze(0)


if __name__ == "__main__":
    query_obj = torch.randn(100, 64, 32)  # [query_num, B, D]
    decoder = TransformerDecoder(6, 32, norm_pre=False)
    memory = torch.rand(50, 64, 32)  # [T, B, D]
    query_pos = torch.rand(query_obj.shape)
    pos = torch.rand(memory.shape)
    out = decoder(query_obj, memory, query_pos=query_pos, pos=pos)
    # print(query_obj.shape, query_obj[0][0])
    print(out.shape)

