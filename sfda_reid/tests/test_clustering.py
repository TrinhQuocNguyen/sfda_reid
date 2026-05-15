import numpy as np
import pytest
from sfda_reid.clustering.dbscan_cluster import DBSCANClustering
from sfda_reid.clustering.camera_aware_refinement import CameraAwareLabelRefinement

def test_dbscan_no_noise_points():
    X = np.vstack([np.random.randn(10, 4) + i*5 for i in range(3)])
    clusterer = DBSCANClustering(eps=3.0, min_samples=2)
    labels, _ = clusterer.fit(X)
    assert (labels != -1).all()

def test_camera_refinement_reduces_noise():
    X = np.vstack([np.random.randn(10, 4) + i*5 for i in range(3)])
    labels = np.array([i for i in range(3) for _ in range(10)])
    # Inject 10% noise
    n_noise = int(0.1 * len(labels))
    labels[:n_noise] = -1
    cam_ids = np.random.randint(0, 3, size=len(labels))
    refiner = CameraAwareLabelRefinement(camera_weight=0.3)
    refined = refiner.refine_labels(X, labels, cam_ids)
    assert (refined != -1).sum() >= (labels != -1).sum()

def test_camera_cooccurrence_matrix_symmetry():
    cam_ids = np.array([0, 1, 2, 1, 0, 2])
    refiner = CameraAwareLabelRefinement()
    mat = refiner.build_camera_cooccurrence_matrix(cam_ids)
    assert np.allclose(mat, mat.T)

def test_label_noise_rate_estimation():
    X = np.random.randn(100, 8)
    labels = np.array([0]*40 + [1]*40 + [-1]*20)
    refiner = CameraAwareLabelRefinement()
    noise = refiner.estimate_label_noise_rate(X, labels)
    assert 0.15 <= noise <= 0.25
