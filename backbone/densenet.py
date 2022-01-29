from typing import List, Tuple
import torch
from torch import nn, Tensor

class TransitionLayer(nn.Module):
    def __init__(self, inChannels: int, numGroups: int):
        super(TransitionLayer, self).__init__()

        self.outChannels = int(inChannels / 2)

        self.module = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, self.outChannels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, numGroups: int, growthRate: int):
        super(ConvBlock, self).__init__()

        self.outChannels = growthRate

        self.module = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, 4 * growthRate, 1),
            nn.GroupNorm(numGroups, 4 * growthRate),
            nn.ReLU(),
            nn.Conv2d(4 * growthRate, growthRate, 3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class DenseBlock(nn.Module):
    def __init__(self, inChannels: int, numBlocks: int, numGroups: int, growthRate: int):
        super(DenseBlock, self).__init__()

        self.outChannels = inChannels

        self.layers = nn.ModuleList()
        for _ in range(numBlocks):
            self.layers.append(ConvBlock(self.outChannels, numGroups, growthRate))
            self.outChannels += growthRate

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, numGroups: int, growthRate: int, numBlocks: List[int]):
        super(DenseNet, self).__init__()

        self.outChannels = 64

        self.input = nn.Sequential(
            nn.Conv2d(3, self.outChannels, 7, padding=3),
            nn.GroupNorm(numGroups, self.outChannels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layers = [self.input]

        for blocks in numBlocks:
            block = DenseBlock(self.outChannels, blocks, numGroups, growthRate)
            self.outChannels = block.outChannels
            trans = TransitionLayer(self.outChannels, numGroups)
            self.outChannels = trans.outChannels
            layers.append(block)
            layers.append(trans)

        self.module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

if __name__ == "__main__":
    
    x = torch.randn(32, 64, 224, 224)
    print(x.shape)
    inChannels, numBlocks, numGroups, growthRate = 64, 5, 8, 64
    print("inChannels, numBlocks, numGroups, growthRate = 64, 5, 8, 64")
    denseblock = DenseBlock(inChannels, numBlocks, numGroups, growthRate)
    print(denseblock(x).shape)
