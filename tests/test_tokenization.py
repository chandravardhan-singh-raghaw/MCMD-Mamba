"""Tests for tokenization (downsample, flatten, project, shape)."""

import torch
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Tokenizer is not yet implemented; placeholder test for shape contract
def test_tokenizer_contract():
    """When implemented: input (B, C, H, W) -> output (B, N, D)."""
    B, C, H, W = 2, 64, 24, 24
    x = torch.randn(B, C, H, W)
    # After tokenization: N = (H/p) * (W/p), D = d_model
    # assert out.shape == (B, N, D)
    assert x.shape == (B, C, H, W)
