"""Local quick smoke tests for Stage 2 (scans, MDSSM, block, stack). Run from repo root: PYTHONPATH=src python -m mcmd_mamba.models.stage2.tests_stage2"""

import torch

from .scans import (
    horizontal_indices,
    vertical_indices,
    spiral_indices_topright_ccw,
    apply_index,
    invert_index,
)
from .mamba_core import build_core, assert_core_io
from .md_ssm import MDSSM, split_concat_sequence, merge_concat_sequence, assert_mdssm_io
from .md_mamba_block import MDMambaBlock, FFN, residual_add, assert_block_io
from .stack import MDMambaStack


def test_scans_roundtrip():
    h, w = 4, 5
    B, D = 2, 8
    x = torch.randn(B, h * w, D)
    for name, idx_fn in [
        ("horizontal", horizontal_indices),
        ("vertical", vertical_indices),
        ("spiral_ccw", spiral_indices_topright_ccw),
    ]:
        ind = idx_fn(h, w)
        inv = invert_index(ind)
        y = apply_index(x, ind, dim=1)
        z = apply_index(y, inv, dim=1)
        assert torch.allclose(x, z), f"{name} roundtrip failed"
    print("  scans roundtrip OK")


def test_mdssm_shape():
    B, L, D = 2, 12, 16
    h, w = 3, 4
    assert h * w == L
    x = torch.randn(B, 3 * L, D)
    core = build_core("dummy", D)
    m = MDSSM(d_model=D, token_hw=(h, w), core=core, conv_kernel=3, merge="sum")
    y = m(x)
    assert y.shape == x.shape, f"MDSSM {y.shape} != {x.shape}"
    assert_mdssm_io(x, y, L, D)
    print("  MDSSM shape OK")


def test_mdmamba_block_shape():
    B, L, D = 2, 12, 16
    h, w = 3, 4
    x = torch.randn(B, 3 * L, D)
    core = build_core("dummy", D)
    md_ssm = MDSSM(d_model=D, token_hw=(h, w), core=core, conv_kernel=3, merge="sum")
    block = MDMambaBlock(d_model=D, md_ssm=md_ssm, dropout=0.0)
    y = block(x)
    assert y.shape == x.shape, f"Block {y.shape} != {x.shape}"
    assert_block_io(x, y, D)
    print("  MDMambaBlock shape OK")


def test_stack_shape():
    B, L, D = 2, 12, 16
    h, w = 3, 4
    x = torch.randn(B, 3 * L, D)
    stack = MDMambaStack(d_model=D, token_hw=(h, w), num_blocks=3, dropout=0.0)
    y = stack(x)
    assert y.shape == x.shape, f"Stack {y.shape} != {x.shape}"
    print("  MDMambaStack shape OK")


def test_split_merge_roundtrip():
    B, L, D = 2, 6, 8
    x = torch.randn(B, 3 * L, D)
    xs, xc, xf = split_concat_sequence(x, L)
    y = merge_concat_sequence(xs, xc, xf)
    assert torch.allclose(x, y)
    print("  split_concat_sequence / merge_concat_sequence OK")


def main():
    torch.manual_seed(42)
    print("Stage2 smoke tests")
    test_scans_roundtrip()
    test_split_merge_roundtrip()
    test_mdssm_shape()
    test_mdmamba_block_shape()
    test_stack_shape()
    print("Stage2 smoke tests passed.")


if __name__ == "__main__":
    main()
