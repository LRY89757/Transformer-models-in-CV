from dot_product import scaled_dot_product
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, output_dim, heads):
        '''
        input_dim and the output_dim must be equal there. we want to use residual connection after.
        heads is the number of the heads
        '''
        super().__init__()
        self.h = heads

        self.fc1 = nn.Linear(input_dim, input_dim*3)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.init_weights()
    

    def init_weights(self):
        
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        '''
        x's shape: [B, T, Dk], Dk means the input_dim.
        '''
        H = self.h
        B, T, D = x.size()

        qkv = self.fc1(x) # [B, T, 3 * H * (D/H)]
        qkv = qkv.reshape(B, T, H, int(D/H), 3)  # [B, T, H, D/H, 3]
        qkv = qkv.permute(0, 2, 1, 3, 4) # [B, H, T, D/H, 3]
        q, k, v = qkv.chunk(3, dim=-1)  # q(as same as k, v): [B, H, T, D/H]

        q, k, v = map(torch.squeeze, [q, k, v])
        att = scaled_dot_product(q, k, v)  # [B, H, T, D/H]
        att = att.permute(0, 2, 1, 3) # [B, T, H, D/H]
        att = att.reshape(B, T, -1) # [B, T, D]
        out = self.fc2(att) # [B, T, D]
        out = att

        return out


if __name__ == "__main__":
    input_dim, output_dim, heads = 512, 512, 8
    multi = MultiHeadAttention(input_dim, output_dim, heads)
    X = torch.randn(16, 30, 512)
    out = multi(X)
    print(out.shape)





