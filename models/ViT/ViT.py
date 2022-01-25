import torch
from torch import nn
import sys,os
sys.path.append(os.getcwd())
from utils.multi_head import MultiHeadAttention
from utils.img_to_patch import Img2Patch


class PreLN_Block(nn.Module):

    def __init__(self, input_dim, output_dim, heads, dropout=0, mlp_dim=512):
        super().__init__()
        self.multi = MultiHeadAttention(input_dim, output_dim, heads)

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, output_dim),
            nn.Dropout()
        )

    def forward(self, x):
        multi = self.multi(self.ln1(x))
        x = x + multi

        mlp = self.mlp(self.ln2(x))
        out = mlp + x

        return out



class VisionTransformer(nn.Module):

    def __init__(self, patch_size, channels, num_classes,  
    L_loops,  embed_dim, heads, dropout=0, mlp_dim=512):
        super().__init__()
        
        input_dim = channels * patch_size * patch_size
        num_patches = input_dim

        self.im2patch = Img2Patch(kernel_size=patch_size, stride=patch_size)

        self.Linear_proj = nn.Linear(input_dim, embed_dim)

        self.encoder = nn.Sequential(*[PreLN_Block(embed_dim, embed_dim, heads, dropout=0, mlp_dim=mlp_dim) for i in range(L_loops)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+num_patches,embed_dim))

    def forward(self, x):
        '''
        x's shape: [B, C, H, W]
        '''
        patches = self.im2patch(x)  # [B, H * W / (patch_size**2), patch_size*patch_size*channels] = [B, T, I]
        B, T, _ = patches.shape
        x = self.Linear_proj(patches) # [B, T, D]

         # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:,:T+1]

        # apply the encoder
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.encoder(x)

        # Perform classification prediction
        cls = x[0]
        out = self.mlp_head(cls)
        return out



if __name__ == "__main__":
    x = torch.randn(64, 3, 224, 224)

    VIT = VisionTransformer(patch_size=16, channels=3, num_classes=10, L_loops=8, embed_dim=512, heads=8)

    out = VIT(x)

    print(out.shape)



