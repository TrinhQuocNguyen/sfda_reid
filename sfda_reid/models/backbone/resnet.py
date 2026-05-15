import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple

class ResNet50Backbone(nn.Module):
    """
    ResNet-50 backbone for ReID with BNNeck and optional classifier.
    Removes final FC, adds global average pooling and BNNeck.
    """
    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes) if num_classes > 0 else None
        nn.init.normal_(self.bnneck.weight, 1.0, 0.02)
        nn.init.constant_(self.bnneck.bias, 0.0)
        if self.classifier:
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.base(x)
        pooled = self.gap(feat).view(x.size(0), -1)
        bn_feat = self.bnneck(pooled)
        logits = self.classifier(bn_feat) if self.classifier else None
        return bn_feat, logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.base(x)
        pooled = self.gap(feat).view(x.size(0), -1)
        return pooled
