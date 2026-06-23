"""Tests for weighted fusion (split, AP/MP, learnable weights)."""

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcmd_mamba.models.stage3.weighted_fusion import WeightedFusion


def test_weighted_fusion_weights_sum_to_one():
    """When forward is implemented: optional check that softmax(weights) sums to 1."""
    cfg = {"pool_modes": ["avg", "max"], "learnable_weights": True}
    m = WeightedFusion(cfg, d_model=64)
    assert m.weights is not None
    assert m.weights.shape == (2,)
    w = torch.softmax(m.weights, dim=0)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
