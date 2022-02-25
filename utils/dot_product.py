import torch
import torch.nn.functional as F
import math

def scaled_dot_product(Q, K, V, return_att=False, bias=None, mask=None):
    '''
    Input: Q,K,V shape may be [Batch, Heads, T, Dk], 
            Dk is the dim of the token, T is the length of the sequence.
           bias: default None, if not None, it will be the position bias added to the qk^T
           mask: default None, if not None, it will be the mask code for swin transformer.
    output: [B, H, T, Dk], [B, H, T, T]
    '''
    Dk = Q.size()[-1]
    K_T = K.transpose(-1, -2)
    if bias is None:   
        att_weight = F.softmax(torch.matmul(Q, K_T) / math.sqrt(Dk), dim=-1)
    elif mask is None:
        att_weight = F.softmax(torch.matmul(Q, K_T) / math.sqrt(Dk) + bias, dim=-1)
    else:
        att_weight = F.softmax(torch.matmul(Q, K_T) / math.sqrt(Dk) + bias + mask, dim=-1)
    mean_values = torch.matmul(att_weight, V)

    if return_att:
        return mean_values, att_weight
    else:
        return mean_values


def dot_product(Q, K, V, return_att=False):
    '''
    Input: Q,K,V shape may be [Batch, Heads, T, Dk], 
            Dk is the dim of the token, T is the length of the sequence.
    output: [B, H, T, Dk], [B, H, T, T]
    '''
    K_T = K.transpose(-1, -2)
    att_weight = F.softmax(torch.matmul(Q, K_T), dim=-1)
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
