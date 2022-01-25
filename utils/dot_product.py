import torch
import torch.nn.functional as F
import math

def scaled_dot_product(Q, K, V, return_att=False):
    '''
    Input: Q,K,V shape may be [Batch, Heads, T, Dk], 
            Dk is the dim of the token, T is the length of the sequence.
    output: [B, H, T, Dk], [B, H, T, T]
    '''
    Dk = Q.size()[-1]
    K_T = K.transpose(-1, -2)
    att_weight = F.softmax(torch.matmul(Q, K_T) / math.sqrt(Dk), dim=-1)
    mean_values = torch.matmul(att_weight, V)

    if return_att:
        return mean_values, att_weight
    else:
        return mean_values

if __name__ == "__main__":
    q = torch.randn(8, 4, 10, 8)
    k = torch.randn(8, 4, 10, 8)
    v = torch.randn(8, 4, 10, 8)

    m_v, a_w = scaled_dot_product(q, k, v, return_att=True)
    print(m_v.shape, a_w.shape)
