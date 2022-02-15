import torch
from torch import nn
from torch import Tensor
from transformer import Transformer
import torch.nn.functional as F
from backbone.resnet import resnet50

# self, embed_dim, num_heads:int=8, activ="relu",
# dropout=0.1, mlp_dim=256, norm_pre=True, num_encoderlayers=6, num_decoderlayers=6):


def build_transformer(args):
    return Transformer(
        d_model=args.embed_dim,
        dropout=args.dropout,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoderlayers=args.enc_layers,
        num_decoderlayers=args.dec_layers,
        norm_pre=args.pre_norm,
    )


class mlp(nn.module):
    """ very simple multi-layer perceptron (also called ffn)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.modulelist(nn.linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, args):
        self.backbone = resnet50(pretrained=True)
        self.reshape = nn.Conv2d(512, args.embed_dim, 1)
        self.transform = build_transformer(args)
        self.bbox_embed = mlp(args.embed_dim, args.embed_dim, 4, 3)
        self.classes_embed = nn.Linear(args.embed_dim, args.Classnum)
        self.query_pos = nn.Embedding(args.querynums, args.embed_dim)

    def forward(self, x: Tensor):
        ''''
        x's shape: [B, 3, H, W]
        '''
        features = self.backbone(x)
        features = self.reshape(features)

        trans, _ = self.transform(features, None, self.query_pos.weight)

        outputsClass = self.classes_embed(trans)
        outputsCoord = self.bbox_embed(trans).sigmoid()

        return {'class': outputsClass[-1],
                'bbox': outputsCoord[-1],
                'aux': [{'class': oc, 'bbox': ob} for oc, ob in zip(outputsClass[:-1], outputsCoord[:-1])]}
