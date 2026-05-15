import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraInvarianceLoss(nn.Module):
    """
    Cross-entropy loss for camera-invariance branch with label smoothing.
    """
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        smooth_labels = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        loss = -torch.sum(F.log_softmax(logits, dim=1) * smooth_labels, dim=1).mean()
        return loss
