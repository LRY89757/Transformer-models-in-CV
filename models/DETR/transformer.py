import sys,os
sys.path.append(os.getcwd())
from models.Encoder.encoder_detr import TransformerEncoder
from models.Decoder.decoder_detr import TransformerDecoder
import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads:int=8, activ="relu", 
        dropout=0.1, mlp_dim=256, norm_pre=True, num_encoderlayers=6, num_decoderlayers=6):
        super().__init__()
        self.encoder = TransformerEncoder(num_encoderlayers, embed_dim, num_heads, mlp_dim, activ, dropout, norm_pre)
        self.decoder = TransformerDecoder(num_decoderlayers, embed_dim, num_heads, activ, dropout, mlp_dim, norm_pre)

    def forward(self, src, src_pos, query_pos):
        '''
        src: [B, C, H, W]
        src_pos: [B, C, H, W]
        query_pos:[query_num, D]
        '''
        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1) # [T, B, D]
        src_pos = src_pos.flatten(2).permute(2, 0, 1) # [T, B, D]
        query_pos = query_pos.unsqueeze(1).repeat(1, B, 1) # [query_num, B, D]
        query = torch.zeros_like(query_pos)

        memory = self.encoder(src, src_pos)   # [T, B, D]
        out = self.decoder(query, memory, query_pos, src_pos) # [query_num, B, D]

        return out.transpose(1, 2), memory.permute(1, 2, 0).view(B, C, H, W)
        # return out.permute(1, 0, 2), memory.permute(1, 2, 0).view(B, C, H, W)


if __name__ == "__main__":
    src = torch.rand(32, 512, 28, 28)  # [B, C, H, W]
    src_pos = torch.zeros_like(src)
    query_pos = torch.rand(100, 512) # [query_num, D]
    transformer = Transformer(512, 8)
    out, memo = transformer(src, src_pos, query_pos)
    print("transformer's decoder shape:", out.shape)
# transformer's decoder shape: torch.Size([1, 32, 100, 512])
    print("transformer's encoder shape:", memo.shape)
# transformer's encoder shape: torch.Size([32, 512, 28, 28])