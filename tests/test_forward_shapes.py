"""Tests for forward pass tensor shapes (placeholder until model is wired)."""

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcmd_mamba.models.mcmd_mamba import MCMDMamba


def test_mcmd_mamba_placeholder():
    """Top-level model exists; forward not implemented yet."""
    cfg = {"model": {"use_mamba": False, "use_weighted_fusion": False}}
    model = MCMDMamba(cfg, num_classes=5)
    assert model.num_classes == 5
    with pytest.raises(NotImplementedError):
        model.forward(torch.randn(2, 3, 384, 384))
