import logging
from typing import List, Tuple

import numpy as np

class DBSCANClustering:
    """
    DBSCAN clustering with FAISS fallback for noise points.
    """
    def __init__(self, eps: float, min_samples: int, use_faiss: bool = True):
        self.eps = eps
        self.min_samples = min_samples
        self.use_faiss = use_faiss
        self.logger = logging.getLogger("DBSCANClustering")

    def _pairwise_distances(self, features: np.ndarray) -> np.ndarray:
        squared_norms = np.sum(features * features, axis=1, keepdims=True)
        distances = squared_norms + squared_norms.T - 2.0 * features @ features.T
        return np.sqrt(np.clip(distances, a_min=0.0, a_max=None))

    def _region_query(self, distmat: np.ndarray, index: int) -> np.ndarray:
        return np.where(distmat[index] <= self.eps)[0]

    def _expand_cluster(
        self,
        distmat: np.ndarray,
        labels: np.ndarray,
        visited: np.ndarray,
        seed_index: int,
        neighbors: np.ndarray,
        cluster_id: int,
    ) -> None:
        labels[seed_index] = cluster_id
        search_queue: List[int] = neighbors.tolist()
        queue_index = 0

        while queue_index < len(search_queue):
            neighbor_index = search_queue[queue_index]
            if not visited[neighbor_index]:
                visited[neighbor_index] = True
                neighbor_neighbors = self._region_query(distmat, neighbor_index)
                if neighbor_neighbors.size >= self.min_samples:
                    for candidate in neighbor_neighbors.tolist():
                        if candidate not in search_queue:
                            search_queue.append(candidate)
            if labels[neighbor_index] == -1:
                labels[neighbor_index] = cluster_id
            queue_index += 1

    def _fit_dbscan(self, features: np.ndarray) -> np.ndarray:
        num_samples = features.shape[0]
        if num_samples == 0:
            return np.empty((0,), dtype=np.int64)

        distmat = self._pairwise_distances(features)
        labels = np.full(num_samples, -1, dtype=np.int64)
        visited = np.zeros(num_samples, dtype=bool)
        cluster_id = 0

        for sample_index in range(num_samples):
            if visited[sample_index]:
                continue
            visited[sample_index] = True
            neighbors = self._region_query(distmat, sample_index)
            if neighbors.size < self.min_samples:
                continue
            self._expand_cluster(distmat, labels, visited, sample_index, neighbors, cluster_id)
            cluster_id += 1

        return labels

    def _assign_noise_to_centroids(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        noise_indices = np.where(labels == -1)[0]
        if noise_indices.size == 0:
            return labels

        unique_labels = np.unique(labels[labels != -1])
        if unique_labels.size == 0:
            return labels

        centroids = self.compute_cluster_centroids(features, labels)
        if centroids.size == 0:
            return labels

        diff = features[noise_indices, None, :] - centroids[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        nearest = np.argmin(distances, axis=1)
        labels[noise_indices] = unique_labels[nearest]
        return labels

    def fit(self, features: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run DBSCAN on L2-normalized features.
        Returns pseudo-labels array of shape (N,).
        Noise points (label == -1) are reassigned to the nearest cluster
        centroid using a NumPy nearest-centroid fallback when clusters exist.

        Args:
            features: Feature matrix with shape (N, D).

        Returns:
            Tuple containing pseudo-labels with shape (N,) and the number of
            non-noise clusters discovered.
        """
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        labels = self._fit_dbscan(features)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if num_clusters > 0:
            labels = self._assign_noise_to_centroids(features, labels)
        noise_ratio = (labels == -1).sum() / len(labels)
        self.logger.info(f"DBSCAN: {num_clusters} clusters, noise ratio {noise_ratio:.3f}")
        return labels, num_clusters

    def compute_cluster_centroids(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Compute one centroid per non-noise cluster.

        Args:
            features: Feature matrix with shape (N, D).
            labels: Integer cluster labels with shape (N,).

        Returns:
            Array with shape (num_clusters, D). Returns an empty array when no
            valid clusters are available.
        """
        unique_labels = np.unique(labels[labels != -1])
        if unique_labels.size == 0:
            return np.empty((0, features.shape[1]), dtype=features.dtype)
        centroids = np.stack([
            features[labels == l].mean(axis=0) for l in unique_labels
        ], axis=0)
        return centroids
