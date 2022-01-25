import sys,os
sys.path.append(os.getcwd())
from utils.multi_head import MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, output_dim, heads, dropout=0, mlp_dim=512):
        super().__init__()

        self.multi = MultiHeadAttention(input_dim, output_dim, heads)

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, output_dim),
            nn.Dropout()
        )
        self.activ = nn.ReLU(inplace=True)


    def forward(self, x):
        mul = self.multi(x)
        x = self.ln1(x + mul)

        mlp = self.mlp(x)
        out = self.ln2(x + mlp)

        return out



class TransformerEncoder(nn.Module):

    def __init__(self, N_loops, input_dim, output_dim, heads, dropout=0, mlp_dim=512):
        super().__init__()
        self.model = nn.Sequential(*[EncoderBlock(input_dim, output_dim, heads, dropout=0, mlp_dim=mlp_dim) for i in range(N_loops)])

    def forward(self, x):
        return self.model(x) 





if __name__ == "__main__":
    x = torch.randn(32, 60, 32) # [B, T, D]
    input_dim, output_dim, heads = 32, 32, 4
    encoder = TransformerEncoder(N_loops=8, input_dim=input_dim, output_dim=output_dim, heads=heads)
    print(encoder(x).shape)



