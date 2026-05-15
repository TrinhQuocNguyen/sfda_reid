import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict
from ..models.sfda_reid import SFDAReID
from ..clustering.dbscan_cluster import DBSCANClustering
from ..clustering.camera_aware_refinement import CameraAwareLabelRefinement
from ..engine.evaluator import ReIDEvaluator
from ..theory.bound_estimator import AdaptationBoundEstimator
import logging

class SFDATrainer:
    def __init__(self, model: SFDAReID, target_loader: DataLoader, query_loader: DataLoader, gallery_loader: DataLoader, cfg):
        self.model = model
        self.target_loader = target_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.cfg = cfg
        self.logger = logging.getLogger("adapt")
        self.device = torch.device(cfg.device)
        self.evaluator = ReIDEvaluator()
        self.bound_estimator = AdaptationBoundEstimator(cfg.theory.bound_delta)

    def run(self) -> None:
        for epoch in range(self.cfg.target.num_epochs):
            pseudo_labels = self._run_clustering(epoch)
            losses = self._train_one_epoch(epoch, pseudo_labels)
            if (epoch + 1) % self.cfg.eval_every == 0:
                metrics = self._evaluate(epoch)
                self._compute_and_log_bound(epoch, pseudo_labels)
                self.logger.info(f"Epoch {epoch+1}: mAP={metrics['mAP']:.2f}, Rank-1={metrics['rank1']:.2f}")

    def _run_clustering(self, epoch: int) -> np.ndarray:
        # 1. Extract features for all target training images
        features, cam_ids = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in self.target_loader:
                images = batch[0].to(self.device)
                cams = batch[1].cpu().numpy()
                feats = self.model.forward_inference(images).cpu().numpy()
                features.append(feats)
                cam_ids.append(cams)
        features = np.concatenate(features, axis=0)
        cam_ids = np.concatenate(cam_ids, axis=0)
        # 2. Run DBSCANClustering
        clusterer = DBSCANClustering(eps=self.cfg.clustering.eps, min_samples=self.cfg.clustering.min_samples)
        initial_labels, _ = clusterer.fit(features)
        # 3. Run CameraAwareLabelRefinement
        refiner = CameraAwareLabelRefinement(camera_weight=self.cfg.clustering.camera_weight)
        refined_labels = refiner.refine_labels(features, initial_labels, cam_ids)
        # 4. Assign pseudo-labels to dataset (assume dataset has set_pseudo_labels)
        if hasattr(self.target_loader.dataset, 'set_pseudo_labels'):
            self.target_loader.dataset.set_pseudo_labels(refined_labels)
        # 5. Rebuild DataLoader if needed (not shown)
        # 6. Return pseudo-labels array
        return refined_labels

    def _train_one_epoch(self, epoch: int, pseudo_labels: np.ndarray) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        for batch in self.target_loader:
            images, camids, indices = batch
            images = images.to(self.device)
            camids = camids.to(self.device)
            indices = indices.to(self.device)
            # Assume pseudo_labels are aligned with indices
            labels = torch.tensor(pseudo_labels[indices.cpu().numpy()], dtype=torch.long, device=self.device)
            out = self.model(images, labels, camids, indices)
            loss = out['loss_contrastive']
            loss.backward()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.target_loader)
        return {'loss_total': avg_loss}

    def _evaluate(self, epoch: int) -> Dict[str, float]:
        metrics = self.evaluator.evaluate(self.model, self.query_loader, self.gallery_loader)
        return metrics

    def _compute_and_log_bound(self, epoch: int, pseudo_labels: np.ndarray) -> None:
        eps_s = 0.1
        d_h = 0.2
        lambda_joint = 0.05
        bank_size = self.cfg.memory_bank.size
        num_samples = len(pseudo_labels)
        bound_dict = self.bound_estimator.compute_full_bound(eps_s, d_h, lambda_joint, bank_size, num_samples)
        self.logger.info(
            "Epoch %d: theoretical_bound=%.3f eps_s=%.3f d_h=%.3f lambda=%.3f slack=%.3f",
            epoch + 1,
            bound_dict['bound'],
            bound_dict['eps_s'],
            bound_dict['d_h'],
            bound_dict['lambda'],
            bound_dict['slack'],
        )
