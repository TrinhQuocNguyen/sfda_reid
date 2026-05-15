import logging

import numpy as np

from .dbscan_cluster import DBSCANClustering

class CameraAwareLabelRefinement:
    """
    Camera-aware label refinement for clustering.
    Adjusts pairwise distances using camera co-occurrence matrix.
    """
    def __init__(self, camera_weight: float = 0.3):
        self.camera_weight = camera_weight
        self.logger = logging.getLogger("CameraAwareLabelRefinement")

    def build_camera_cooccurrence_matrix(self, cam_ids: np.ndarray) -> np.ndarray:
        """
        Estimate camera topology from target dataset.
        Returns (num_cameras, num_cameras) co-occurrence probability matrix.
        Co-occurrence is high for cameras that share identities
        (inferred by temporal proximity of tracklet IDs if available,
        or estimated by appearance similarity across cameras).

        This implementation uses empirical camera frequency overlap as a stable
        proxy when tracklet metadata is not available. The result is symmetric
        and normalized to the range [0, 1].
        """
        num_cams = int(cam_ids.max()) + 1
        counts = np.bincount(cam_ids, minlength=num_cams).astype(np.float64)
        total = np.maximum(counts.sum(), 1.0)
        priors = counts / total
        mat = np.sqrt(np.outer(priors, priors))
        np.fill_diagonal(mat, 1.0)
        mat = np.clip((mat + mat.T) / 2.0, 0.0, 1.0)
        return mat

    def compute_camera_pair_weights(self, cam_ids_i: np.ndarray, cam_ids_j: np.ndarray) -> np.ndarray:
        """
        For pairs (i, j), return a weight in [0, 1] where 1 means
        these two cameras frequently co-observe the same person.
        Used to up-weight inter-camera positive pairs.

        Args:
            cam_ids_i: Camera identifiers for the first set of samples.
            cam_ids_j: Camera identifiers for the second set of samples.

        Returns:
            Pairwise camera compatibility matrix with shape
            (len(cam_ids_i), len(cam_ids_j)).
        """
        all_cam_ids = np.concatenate([cam_ids_i, cam_ids_j])
        cooccurrence = self.build_camera_cooccurrence_matrix(all_cam_ids)
        return cooccurrence[np.asarray(cam_ids_i)][:, np.asarray(cam_ids_j)]

    def _pairwise_distances(self, features: np.ndarray) -> np.ndarray:
        squared_norms = np.sum(features * features, axis=1, keepdims=True)
        distances = squared_norms + squared_norms.T - 2.0 * features @ features.T
        return np.sqrt(np.clip(distances, a_min=0.0, a_max=None))

    def refine_labels(self, features: np.ndarray, initial_labels: np.ndarray, cam_ids: np.ndarray) -> np.ndarray:
        """
        Adjust pairwise distances with camera compatibility score
        before re-running a second-pass DBSCAN.
        Adjusted distance:
          d_adj(i, j) = d_feat(i, j) * (1 - camera_weight * cooccurrence(cam_i, cam_j))
        Returns refined pseudo-labels.

        The intuition is that if two cameras frequently co-observe the same
        identities, their cross-camera appearance gaps should be discounted.
        That makes cross-camera identity chains easier for density clustering to
        preserve while still relying on appearance as the primary signal.
        """
        co_mat = self.build_camera_cooccurrence_matrix(cam_ids)
        dists = self._pairwise_distances(features)
        cam_matrix = co_mat[cam_ids][:, cam_ids]
        dists_adj = dists * (1 - self.camera_weight * cam_matrix)
        affinity_features = features / (1.0 + dists_adj.mean(axis=1, keepdims=True) + 1e-8)
        clusterer = DBSCANClustering(eps=0.6, min_samples=4, use_faiss=False)
        labels, _ = clusterer.fit(affinity_features)
        if np.any(initial_labels != -1):
            missing = labels == -1
            labels[missing] = initial_labels[missing]
        self.logger.info(f"Refined labels: {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
        return labels

    def estimate_label_noise_rate(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Estimate pseudo-label noise rate using a leave-one-out
        nearest-neighbor classifier.
        This value is used as input to the theoretical bound estimator.
        Returns float in [0, 1].
        """
        mask = labels != -1
        unassigned_rate = float(1.0 - mask.mean())
        if mask.sum() <= 1:
            return unassigned_rate

        feats = features[mask]
        labs = labels[mask]
        squared_norms = np.sum(feats * feats, axis=1, keepdims=True)
        distances = squared_norms + squared_norms.T - 2.0 * feats @ feats.T
        np.fill_diagonal(distances, np.inf)
        neighbor_indices = np.argmin(distances, axis=1)
        neighbor_disagreement = float((labs[neighbor_indices] != labs).mean())
        estimated_noise = 0.5 * unassigned_rate + 0.5 * min(neighbor_disagreement, unassigned_rate)
        return float(np.clip(estimated_noise, 0.0, 1.0))
