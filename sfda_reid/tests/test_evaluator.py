import numpy as np
import torch
from sfda_reid.engine.evaluator import ReIDEvaluator

def test_perfect_retrieval_gives_map_1():
    evaluator = ReIDEvaluator()
    feats = torch.eye(5)
    pids = [0, 1, 2, 3, 4]
    q_camids = [0, 0, 0, 0, 0]
    g_camids = [1, 1, 1, 1, 1]
    distmat = (feats[:, None, :] - feats[None, :, :]).pow(2).sum(-1).numpy()
    cmc, mAP = evaluator._eval_func(distmat, pids, pids, q_camids, g_camids, max_rank=5)
    assert abs(mAP - 1.0) < 1e-6
    assert abs(cmc[0] - 1.0) < 1e-6

def test_random_retrieval_map_near_chance():
    evaluator = ReIDEvaluator()
    np.random.seed(0)
    distmat = np.random.rand(100, 100)
    pids = list(range(100))
    q_camids = [0] * 100
    g_camids = [1] * 100
    cmc, mAP = evaluator._eval_func(distmat, pids, pids, q_camids, g_camids, max_rank=5)
    assert 0.01 < mAP < 0.08

def test_same_camera_filtering():
    evaluator = ReIDEvaluator()
    distmat = np.zeros((2, 2))
    q_pids = [0, 1]
    g_pids = [0, 1]
    q_camids = [0, 1]
    g_camids = [0, 1]
    cmc, mAP = evaluator._eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=2)
    assert mAP == 0.0
