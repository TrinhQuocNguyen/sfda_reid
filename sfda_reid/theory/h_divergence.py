import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional

class HDivergenceEstimator:
    """
    Estimates the H-divergence between source and target feature distributions
    using a domain discriminator approach (Ben-David et al., 2010).
    Since source data is unavailable at adaptation time, we estimate using
    the source model's output distribution on target features as a proxy.
    """
    def __init__(self, feature_dim: int, num_epochs: int = 10, sample_size: int = 5000):
        self.feature_dim = feature_dim
        self.num_epochs = num_epochs
        self.sample_size = sample_size

    def estimate_from_model_uncertainty(self, model: nn.Module, target_loader: DataLoader) -> float:
        """
        Proxy estimation without source data.
        Uses mean entropy of source model predictions on target samples
        as a surrogate for domain divergence.
        High entropy -> source model is confused -> high divergence.
        Returns estimated d_H in [0, 2].
        """
        model.eval()
        entropies = []
        with torch.no_grad():
            for batch in target_loader:
                images = batch['image'] if isinstance(batch, dict) else batch[0]
                logits = model(images.cuda() if torch.cuda.is_available() else images)
                probs = torch.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropies.append(entropy.cpu().numpy())
        mean_entropy = np.mean(np.concatenate(entropies))
        return float(np.clip(mean_entropy, 0, 2))

    def estimate_with_discriminator(self, source_features: np.ndarray, target_features: np.ndarray) -> float:
        """
        Classic discriminator-based H-divergence estimation.
        Used in bound validation experiments (Experiment A/B)
        where we do have synthetic source features for comparison.
        Trains a binary classifier to distinguish source vs target.
        d_H = 2 * (1 - 2 * min_classification_error)
        Returns float in [0, 2].
        """
        from sklearn.linear_model import LogisticRegression
        X = np.vstack([source_features, target_features])
        y = np.array([0] * len(source_features) + [1] * len(target_features))
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        acc = clf.score(X, y)
        min_err = min(acc, 1 - acc)
        d_h = 2 * (1 - 2 * min_err)
        return float(np.clip(d_h, 0, 2))
