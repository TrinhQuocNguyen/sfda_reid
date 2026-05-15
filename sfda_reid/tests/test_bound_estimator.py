import numpy as np
from sfda_reid.theory.bound_estimator import AdaptationBoundEstimator

def test_memory_bank_slack_decreases_with_bank_size():
    estimator = AdaptationBoundEstimator()
    slack1 = estimator.compute_memory_bank_slack(1000, 100, 0.05)
    slack2 = estimator.compute_memory_bank_slack(2000, 100, 0.05)
    assert slack2 < slack1

def test_full_bound_is_sum_of_components():
    estimator = AdaptationBoundEstimator()
    out = estimator.compute_full_bound(0.1, 0.2, 0.05, 1000, 100)
    expected = out['eps_s'] + out['d_h'] + out['lambda'] + out['slack']
    assert abs(out['bound'] - expected) < 1e-6

def test_bound_nonnegative():
    estimator = AdaptationBoundEstimator()
    out = estimator.compute_full_bound(0.1, 0.2, 0.05, 1000, 100)
    assert out['bound'] >= 0
