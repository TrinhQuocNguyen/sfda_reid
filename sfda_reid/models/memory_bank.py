import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class ContrastiveMemoryBank(nn.Module):
    """
    Memory bank for contrastive learning with EMA update and circular queue.
    """
    def __init__(self, num_features: int, bank_size: int, temperature: float, momentum: float):
        super().__init__()
        self.bank_size = bank_size
        self.num_features = num_features
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer('bank_features', torch.zeros(bank_size, num_features))
        self.register_buffer('bank_labels', torch.full((bank_size,), -1, dtype=torch.long))
        self.ptr = 0

    def forward(self, features: Tensor, labels: Tensor) -> Tensor:
        """
        Compute contrastive loss against memory bank.
        Positive: same pseudo-label. Negative: different pseudo-label.
        Returns scalar loss.
        """
        features = nn.functional.normalize(features, dim=1)
        bank = nn.functional.normalize(self.bank_features, dim=1)
        logits = torch.mm(features, bank.t()) / self.temperature
        mask = labels.unsqueeze(1) == self.bank_labels.unsqueeze(0)
        logits_mask = self.bank_labels.unsqueeze(0) != -1
        logits = logits * logits_mask
        exp_logits = torch.exp(logits)
        pos_mask = mask & logits_mask
        pos_exp = exp_logits * pos_mask
        neg_exp = exp_logits * (~pos_mask & logits_mask)
        pos_sum = pos_exp.sum(dim=1) + 1e-8
        neg_sum = neg_exp.sum(dim=1) + 1e-8
        loss = -torch.log(pos_sum / (pos_sum + neg_sum)).mean()
        return loss

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor, indices: Tensor) -> None:
        """
        Update memory bank slots with EMA of new features.
        features: (B, D), labels: (B,), indices: (B,)
        """
        features = nn.functional.normalize(features, dim=1)
        for feat, label, idx in zip(features, labels, indices):
            idx = idx.item() % self.bank_size
            old_feat = self.bank_features[idx]
            self.bank_features[idx] = self.momentum * old_feat + (1 - self.momentum) * feat
            self.bank_features[idx] = nn.functional.normalize(self.bank_features[idx], dim=0)
            self.bank_labels[idx] = label

    def get_bank_state(self) -> Tuple[Tensor, Tensor]:
        """Returns (bank_features, bank_labels) for inspection."""
        return self.bank_features.clone(), self.bank_labels.clone()
