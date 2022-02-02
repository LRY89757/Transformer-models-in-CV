import torch
from torch import nn
import torch.nn.functional as F
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

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            _getactiv(activ),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim)
        )

        self.drop = nn.Dropout(dropout)

        self.norm_pre = norm_pre
    
    def forward_pre(self, query_obj, memory):
        '''
        ... A lot of param to be add
        query_obj means the input of decoder.
        memory means the output of the encoder.
        '''




