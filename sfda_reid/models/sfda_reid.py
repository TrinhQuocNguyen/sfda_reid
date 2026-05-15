import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict
from .memory_bank import ContrastiveMemoryBank

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class SFDAReID(nn.Module):
    """
    Source-Free Domain Adaptive ReID model with camera-invariance branch.
    """
    def __init__(self, backbone: nn.Module, memory_bank: ContrastiveMemoryBank, num_cameras: int):
        super().__init__()
        self.backbone = backbone
        self.memory_bank = memory_bank
        self.cam_classifier = nn.Linear(2048, num_cameras)

    def forward(self, x: Tensor, labels: Tensor, cam_ids: Tensor, indices: Tensor) -> Dict[str, Tensor]:
        bn_feat, logits = self.backbone(x)
        # Contrastive loss
        loss_contrastive = self.memory_bank(features=bn_feat, labels=labels)
        # Entropy and camera-invariance losses are computed externally
        # Camera-invariance branch
        grl_feat = grad_reverse(bn_feat)
        cam_logits = self.cam_classifier(grl_feat)
        out = {
            'loss_contrastive': loss_contrastive,
            'features': bn_feat,
            'logits': logits,
            'cam_logits': cam_logits
        }
        return out

    def forward_inference(self, x: Tensor) -> Tensor:
        bn_feat, _ = self.backbone(x)
        return nn.functional.normalize(bn_feat, dim=1)
