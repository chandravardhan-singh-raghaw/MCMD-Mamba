"""Tests for spiral/h/v ordering and invertibility."""

import torch
import pytest

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcmd_mamba.models.stage2.scans import (
    horizontal_scan_indices,
    vertical_scan_indices,
    apply_scan,
    invert_scan,
)


def test_horizontal_scan_order():
    h, w = 4, 5
    ind = horizontal_scan_indices(h, w)
    assert ind.shape == (h * w,)
    assert ind[0] == 0
    assert ind[-1] == h * w - 1
    assert torch.all(ind == torch.arange(h * w))


def test_vertical_scan_order():
    h, w = 4, 5
    ind = vertical_scan_indices(h, w)
    assert ind.shape == (h * w,)
    # column-major: 0, 4, 8, 12, 1, 5, ...
    expected = torch.arange(h * w).reshape(w, h).T.reshape(-1)
    assert torch.all(ind == expected)


def test_apply_scan_invert_roundtrip():
    h, w = 3, 4
    ind = horizontal_scan_indices(h, w)
    inv = invert_scan(ind)
    x = torch.randn(2, h * w, 8)
    y = apply_scan(x, ind, dim=1)
    z = apply_scan(y, inv, dim=1)
    assert torch.allclose(x, z)


def test_vertical_invert_roundtrip():
    h, w = 3, 4
    ind = vertical_scan_indices(h, w)
    inv = invert_scan(ind)
    x = torch.randn(2, h * w, 8)
    y = apply_scan(x, ind, dim=1)
    z = apply_scan(y, inv, dim=1)
    assert torch.allclose(x, z)


def test_spiral_scan_order():
    from mcmd_mamba.models.stage2.scans import spiral_scan_indices
    h, w = 3, 4
    ind = spiral_scan_indices(h, w)
    assert ind.shape == (h * w,)
    assert ind.unique().numel() == h * w  # permutation of 0..N-1


def test_spiral_invert_roundtrip():
    from mcmd_mamba.models.stage2.scans import spiral_scan_indices
    h, w = 3, 4
    ind = spiral_scan_indices(h, w)
    inv = invert_scan(ind)
    x = torch.randn(2, h * w, 8)
    y = apply_scan(x, ind, dim=1)
    z = apply_scan(y, inv, dim=1)
    assert torch.allclose(x, z)
