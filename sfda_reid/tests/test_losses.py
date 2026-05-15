import torch
import pytest
from sfda_reid.losses.contrastive import InfoNCELoss
from sfda_reid.losses.entropy import EntropyMinimizationLoss
from sfda_reid.losses.camera_invariance import CameraInvarianceLoss

def test_infonce_loss_positive_pairs():
    loss_fn = InfoNCELoss()
    queries = torch.randn(8, 32)
    keys = torch.randn(8, 32)
    pos_mask = torch.eye(8).bool()
    loss1 = loss_fn(queries, keys, pos_mask)
    queries[0] = keys[0]  # increase similarity
    loss2 = loss_fn(queries, keys, pos_mask)
    assert loss2 < loss1

def test_entropy_loss_uniform_max():
    loss_fn = EntropyMinimizationLoss()
    probs = torch.full((4, 10), 0.1)
    loss = loss_fn(probs)
    assert abs(loss.item() - 2.302) < 0.01

def test_entropy_loss_peaked_min():
    loss_fn = EntropyMinimizationLoss()
    probs = torch.zeros(4, 10)
    probs[:, 0] = 1.0
    loss = loss_fn(probs)
    assert loss.item() < 1e-3

def test_camera_invariance_loss_shape():
    loss_fn = CameraInvarianceLoss()
    logits = torch.randn(6, 3)
    targets = torch.randint(0, 3, (6,))
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0

def test_grl_reverses_gradient():
    from sfda_reid.models.sfda_reid import grad_reverse
    x = torch.ones(1, requires_grad=True)
    y = grad_reverse(x, lambda_=1.0)
    y.backward()
    assert x.grad.item() == -1.0
