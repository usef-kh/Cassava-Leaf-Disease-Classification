import torch.nn as nn
from torchvision import models


class Vgg(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg19_bn(pretrained=True)
        self.lin = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.vgg(x)
        x = self.lin(x)

        return x
