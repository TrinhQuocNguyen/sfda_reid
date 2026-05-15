import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict
from .h_divergence import HDivergenceEstimator

class AdaptationBoundEstimator:
    """
    Computes the theoretical upper bound on target domain ranking error
    for source-free domain adaptive ReID.

    Bound (Theorem 1):
        eps_T <= eps_S + d_H(D_S, D_T) + lambda + C(K, N, delta)

    Where:
        eps_T  = target ranking error (what we want to bound)
        eps_S  = source model's ranking error (estimated from pseudo-labels)
        d_H    = H-divergence (estimated via HDivergenceEstimator)
        lambda = ideal joint error (approximated via source validation)
        C(K, N, delta) = memory bank slack term (Corollary 1)
                   = sqrt(log(1 + N/K) / (2*N)) * log(1/delta)
        delta  = confidence parameter (default 0.05)

    Corollary 1 (Memory Bank Slack):
        Larger K (bank size) reduces C, tightening the bound.
        Validated empirically in experiments/bound_validation.py.
    """
    def __init__(self, delta: float = 0.05):
        self.delta = delta

    def compute_source_error(self, model: torch.nn.Module, source_val_loader: DataLoader) -> float:
        """
        Estimate eps_S using CMC Rank-1 error (1 - Rank1).
        """
        from ..utils.metrics import compute_cmc
        model.eval()
        features, pids, camids = [], [], []
        with torch.no_grad():
            for batch in source_val_loader:
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                feats = model.forward_inference(images.cuda() if torch.cuda.is_available() else images)
                features.append(feats.cpu())
                pids.extend(batch['pid'] if isinstance(batch, dict) else batch[1])
                camids.extend(batch['camid'] if isinstance(batch, dict) else batch[2])
        features = torch.cat(features, dim=0).numpy()
        distmat = np.linalg.norm(features[:, None] - features[None, :], axis=2)
        cmc = compute_cmc(distmat, pids, pids, camids, camids, ranks=[1])
        return float(1.0 - cmc[0])

    def compute_ideal_joint_error(self, source_val_loader: DataLoader, target_loader: DataLoader) -> float:
        """
        Approximates lambda as the minimum achievable error under
        the optimal hypothesis. Approximated as:
        lambda ≈ noise_rate * (1 - noise_rate)
        where noise_rate is from CameraAwareLabelRefinement.
        """
        # Placeholder: should be replaced by actual noise rate estimation
        noise_rate = 0.1
        return float(noise_rate * (1 - noise_rate))

    def compute_memory_bank_slack(self, bank_size: int, num_samples: int, delta: float) -> float:
        """
        C(K, N, delta) = sqrt(log(1 + N / K) / (2 * N)) * log(1 / delta)
        Returns float >= 0.
        """
        if bank_size <= 0 or num_samples <= 0:
            return 0.0
        slack = np.sqrt(np.log1p(num_samples / bank_size) / (2 * num_samples)) * np.log(1 / delta)
        return float(max(slack, 0.0))

    def compute_full_bound(self, eps_s: float, d_h: float, lambda_joint: float, bank_size: int, num_samples: int) -> Dict[str, float]:
        """
        Returns dict:
        {
          'bound': float,          # full upper bound value
          'eps_s': float,
          'd_h': float,
          'lambda': float,
          'slack': float,
          'components': dict       # breakdown for plotting
        }
        """
        slack = self.compute_memory_bank_slack(bank_size, num_samples, self.delta)
        bound = eps_s + d_h + lambda_joint + slack
        return {
            'bound': bound,
            'eps_s': eps_s,
            'd_h': d_h,
            'lambda': lambda_joint,
            'slack': slack,
            'components': {
                'eps_s': eps_s,
                'd_h': d_h,
                'lambda': lambda_joint,
                'slack': slack
            }
        }
