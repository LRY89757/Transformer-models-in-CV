import torch
from torchvision.models import resnet50
from torch import nn, Tensor
from typing import List, Dict

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

nums_classes = len(CLASSES)

class DETR_demo(nn.Module):
    '''
    simple version of detr. the module is designed for batch_size=1.
    '''
    def __init__(self, num_classes:int=91, hidden_dim:int=256, nheads:int=8, 
            encodernums:int=6, decodernums:int=6) -> None:
        super().__init__()

        self.backbone = resnet50()

        self.transitionlayer = nn.Conv2d(2048, hidden_dim, 1)

        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nheads, 
                num_encoder_layers=encodernums, num_decoder_layers=decodernums)
        
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # self.pos_col = nn.Parameter(torch.rand(50, hidden_dim // 2))
        # self.pos_row = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, 
            batch:Tensor) -> Dict[str, Tensor]:

        # features = self.backbone(batch)   # [B, 2048, H, W]
        x = self.backbone.conv1(batch)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)  # [B, 2048, H, W]



        features = self.transitionlayer(features) # [B, 256, H, W]
        B, _, H, W = features.size()

        querypos = self.query_pos.unsqueeze(1)   # please refer to https://pytorch.org/docs/stable/generated/torch.Tensor.repeat.html#torch-tensor-repeat
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        trans = self.transformer(pos + 0.1 *features.flatten(2).permute(2, 0, 1), querypos).transpose(0, 1)  # 还没加位置编码

        return {'pred_logits': self.linear_class(trans),
                'pred_boxes': self.linear_bbox(trans).sigmoid()}






if __name__ == "__main__":
    model = DETR_demo()

    x = torch.rand(1, 3, 224, 224)

    result = model(x)

    for k, v in result.items():
        print(k + "'s shape: ", v.shape)











