import torch.nn as nn
from torchvision import models


class ResNext(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnext = models.resnext50_32x4d(pretrained=True)

        n_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        x = self.resnext(x)

        return x
