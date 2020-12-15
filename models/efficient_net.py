import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.eff_net = EfficientNet.from_pretrained('efficientnet-b0')
        in_features = self.efficient_net._fc.in_features
        self.eff_net._fc = nn.Linear(in_features, 5)

    def forward(self, x):
        x = self.eff_net(x)
        print(x.shape)

        return x
