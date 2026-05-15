import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class EntropyMinimizationLoss(nn.Module):
    """
    Shannon entropy minimization loss for softmax probabilities.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
        return entropy.mean()

    @staticmethod
    def compute_batch_entropy_map(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Returns per-sample entropy for visualization.
        """
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
        return entropy
