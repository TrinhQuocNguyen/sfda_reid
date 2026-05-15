import torch
import torch.nn as nn
from typing import Tuple

class ViTBackbone(nn.Module):
    """
    Vision Transformer (ViT) backbone for ReID with BNNeck.
    Uses timm's vit_small_patch16_224, replaces classifier with BNNeck.
    Supports patch token averaging or CLS token.
    """
    def __init__(self, num_classes: int = 0, pretrained: bool = True, use_patch_avg: bool = False):
        super().__init__()
        import timm

        self.model = timm.create_model('vit_small_patch16_224', pretrained=pretrained)
        self.use_patch_avg = use_patch_avg
        self.model.head = nn.Identity()
        self.bnneck = nn.BatchNorm1d(384)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(384, num_classes) if num_classes > 0 else None
        nn.init.normal_(self.bnneck.weight, 1.0, 0.02)
        nn.init.constant_(self.bnneck.bias, 0.0)
        if self.classifier:
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.model.forward_features(x)
        if self.use_patch_avg:
            pooled = feat[:, 1:, :].mean(dim=1)
        else:
            pooled = feat[:, 0]
        bn_feat = self.bnneck(pooled)
        logits = self.classifier(bn_feat) if self.classifier else None
        return bn_feat, logits

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.forward_features(x)
        if self.use_patch_avg:
            return feat[:, 1:, :].mean(dim=1)
        else:
            return feat[:, 0]

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """Inference mode: returns BNNeck features for evaluation."""
        feat = self.model.forward_features(x)
        if self.use_patch_avg:
            pooled = feat[:, 1:, :].mean(dim=1)
        else:
            pooled = feat[:, 0]
        bn_feat = self.bnneck(pooled)
        return bn_feat
