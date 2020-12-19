import timm
import torch.nn as nn


class EfficientNet(nn.Module):
    def __init__(self, n_class=5, pretrained=True):
        super().__init__()
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
