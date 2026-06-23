"""
Stage 3 tests: split, pooling, weighted fusion, head, wrapper.
Shape tests, gradient flow, determinism smoke tests.
Run from repo root: PYTHONPATH=src python -m mcmd_mamba.models.stage3.tests_stage3
"""

import logging
import sys
from pathlib import Path

import torch

try:
    from mcmd_mamba.models.stage3 import (
        split_scales,
        infer_L,
        dual_pool_tokens,
        WeightedFusion,
        Stage3Head,
        Stage3,
    )
except ImportError:
    _src = Path(__file__).resolve().parent.parent.parent.parent.parent / "src"
    sys.path.insert(0, str(_src))
    from mcmd_mamba.models.stage3 import (
        split_scales,
        infer_L,
        dual_pool_tokens,
        WeightedFusion,
        Stage3Head,
        Stage3,
    )

LOG = logging.getLogger(__name__)


def test_split_scales():
    """split_scales(T, L) -> Ts, Tc, Tf each (B, L, D)."""
    B, L, D = 2, 12, 32
    T = torch.randn(B, 3 * L, D)
    Ts, Tc, Tf = split_scales(T, L)
    assert Ts.shape == (B, L, D) and Tc.shape == (B, L, D) and Tf.shape == (B, L, D)
    assert torch.allclose(T[:, :L], Ts) and torch.allclose(T[:, L : 2 * L], Tc)
    LOG.info("  split_scales OK")


def test_infer_L():
    """infer_L(T) returns L when T.shape[1] = 3*L."""
    T = torch.randn(2, 36, 16)
    L = infer_L(T)
    assert L == 12
    LOG.info("  infer_L OK")


def test_dual_pool_tokens():
    """dual_pool_tokens(Ti) -> (B, 2D)."""
    B, L, D = 2, 12, 32
    Ti = torch.randn(B, L, D)
    Pi = dual_pool_tokens(Ti)
    assert Pi.shape == (B, 2 * D)
    LOG.info("  dual_pool_tokens OK")


def test_weighted_fusion():
    """WeightedFusion(Ts, Tc, Tf) -> F_fuse (B, 2D)."""
    B, L, D = 2, 12, 32
    Ts = torch.randn(B, L, D)
    Tc = torch.randn(B, L, D)
    Tf = torch.randn(B, L, D)
    fusion = WeightedFusion(d_model=D)
    F_fuse = fusion(Ts, Tc, Tf)
    assert F_fuse.shape == (B, 2 * D)
    assert torch.isfinite(F_fuse).all()
    LOG.info("  WeightedFusion OK")


def test_stage3_head():
    """Stage3Head(F_fuse) -> logits (B, num_classes)."""
    B, embed_dim = 2, 64
    num_classes = 5
    F_fuse = torch.randn(B, embed_dim)
    head = Stage3Head(embed_dim=embed_dim, num_classes=num_classes, dropout=0.0)
    logits = head(F_fuse)
    assert logits.shape == (B, num_classes)
    assert torch.isfinite(logits).all()
    LOG.info("  Stage3Head OK")


def test_stage3_wrapper():
    """Stage3(T_out) -> logits (B, num_classes)."""
    B, L, D = 2, 12, 32
    num_classes = 5
    T_out = torch.randn(B, 3 * L, D)
    stage3 = Stage3(
        embed_dim=D,
        num_classes=num_classes,
        tokens_per_scale=L,
        dropout=0.0,
    )
    logits = stage3(T_out)
    assert logits.shape == (B, num_classes)
    assert torch.isfinite(logits).all()
    LOG.info("  Stage3 wrapper OK")


def test_stage3_gradient_flow():
    """Backward through Stage3: gradients flow, no NaN."""
    B, L, D = 2, 12, 32
    num_classes = 5
    T_out = torch.randn(B, 3 * L, D, requires_grad=True)
    stage3 = Stage3(embed_dim=D, num_classes=num_classes, tokens_per_scale=L, dropout=0.0)
    logits = stage3(T_out)
    loss = logits.sum()
    loss.backward()
    assert T_out.grad is not None and T_out.grad.shape == T_out.shape
    assert torch.isfinite(T_out.grad).all()
    LOG.info("  Stage3 gradient flow OK")


def test_stage3_determinism():
    """Stage3 forward is deterministic for same input/seed."""
    torch.manual_seed(42)
    B, L, D = 2, 12, 32
    num_classes = 5
    T_out = torch.randn(B, 3 * L, D)
    stage3 = Stage3(embed_dim=D, num_classes=num_classes, tokens_per_scale=L, dropout=0.0)
    stage3.eval()
    with torch.no_grad():
        logits1 = stage3(T_out)
        logits2 = stage3(T_out)
    assert torch.allclose(logits1, logits2)
    LOG.info("  Stage3 determinism OK")


def run_all(verbose: bool = True):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    torch.manual_seed(42)
    LOG.info("========== Stage 3 tests ==========")
    test_split_scales()
    test_infer_L()
    test_dual_pool_tokens()
    test_weighted_fusion()
    test_stage3_head()
    test_stage3_wrapper()
    test_stage3_gradient_flow()
    test_stage3_determinism()
    LOG.info("========== Stage 3 tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 3 tests")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    run_all(verbose=not args.quiet)
