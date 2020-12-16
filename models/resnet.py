import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnext50_32x4d()

        n_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.resnet(x)

        return x

