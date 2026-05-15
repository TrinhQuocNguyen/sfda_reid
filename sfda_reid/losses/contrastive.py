import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for source training.
    Args:
        temperature: float
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        logits = torch.div(torch.matmul(features, features.t()), self.temperature)
        logits_mask = torch.ones_like(mask, dtype=torch.bool)
        logits_mask.fill_diagonal_(0)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) - 1 + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for memory bank contrastive learning.
    Args:
        temperature: float
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        queries = F.normalize(queries, dim=1)
        keys = F.normalize(keys, dim=1)
        logits = torch.mm(queries, keys.t()) / self.temperature
        exp_logits = torch.exp(logits)
        pos_exp = exp_logits * pos_mask
        neg_exp = exp_logits * (~pos_mask)
        pos_sum = pos_exp.sum(dim=1) + 1e-8
        neg_sum = neg_exp.sum(dim=1) + 1e-8
        loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
        return torch.clamp(loss, min=0.0)
