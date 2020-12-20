import torch.nn as nn
from torchvision import models


class Inception(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.inception_v3(pretrained=True)
        self.lin = nn.Linear(1000, 5)

    def forward(self, x):
        x = self.model(x)
        x = self.lin(x)

        return x
