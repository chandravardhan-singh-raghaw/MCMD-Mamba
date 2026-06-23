"""
Exhaustive Stage 3 tests: split, pooling, weighted fusion, head, wrapper.
Validates shapes, roundtrips, finiteness, gradient flow, determinism.
Run from repo root: PYTHONPATH=src python tests/test_stage3_exhaustive.py
With pytest: pytest tests/test_stage3_exhaustive.py -v
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
    _src = Path(__file__).resolve().parent.parent / "src"
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


# ---------- Split ----------
def test_split_scales():
    """split_scales(T, L) -> Ts, Tc, Tf each (B, L, D); matches T layout."""
    B, L, D = 2, 12, 32
    T = torch.randn(B, 3 * L, D)
    Ts, Tc, Tf = split_scales(T, L)
    assert Ts.shape == (B, L, D) and Tc.shape == (B, L, D) and Tf.shape == (B, L, D)
    assert torch.allclose(T[:, :L], Ts)
    assert torch.allclose(T[:, L : 2 * L], Tc)
    assert torch.allclose(T[:, 2 * L :], Tf)
    LOG.info("  split: split_scales OK")


def test_infer_L():
    """infer_L(T) returns L when T.shape[1] = 3*L."""
    for N in (9, 36, 588):
        T = torch.randn(2, N, 16)
        L = infer_L(T)
        assert L == N // 3, f"infer_L: N={N} -> L={L}"
    LOG.info("  split: infer_L OK")


# ---------- Pooling ----------
def test_dual_pool_tokens():
    """dual_pool_tokens(Ti) -> (B, 2D); avg+max over token dim."""
    B, L, D = 2, 12, 32
    Ti = torch.randn(B, L, D)
    Pi = dual_pool_tokens(Ti)
    assert Pi.shape == (B, 2 * D)
    avg_part = Pi[:, :D]
    max_part = Pi[:, D:]
    assert torch.allclose(avg_part, Ti.mean(dim=1))
    assert torch.allclose(max_part, Ti.max(dim=1).values)
    LOG.info("  pooling: dual_pool_tokens OK")


# ---------- WeightedFusion ----------
def test_weighted_fusion_shapes():
    """WeightedFusion(Ts, Tc, Tf) -> F_fuse (B, 2D)."""
    B, L, D = 2, 12, 32
    Ts = torch.randn(B, L, D)
    Tc = torch.randn(B, L, D)
    Tf = torch.randn(B, L, D)
    fusion = WeightedFusion(d_model=D)
    F_fuse = fusion(Ts, Tc, Tf)
    assert F_fuse.shape == (B, 2 * D)
    assert torch.isfinite(F_fuse).all()
    LOG.info("  weighted_fusion: shapes OK")


def test_weighted_fusion_weights_softmax():
    """WeightedFusion weights sum to 1 (softmax normalized)."""
    fusion = WeightedFusion(d_model=8)
    w = torch.softmax(fusion.weights, dim=0)
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert w.shape == (3,)
    LOG.info("  weighted_fusion: weights softmax OK")


# ---------- Stage3Head ----------
def test_stage3_head():
    """Stage3Head(F_fuse) -> logits (B, num_classes)."""
    B, embed_dim, num_classes = 2, 64, 5
    F_fuse = torch.randn(B, embed_dim)
    head = Stage3Head(embed_dim=embed_dim, num_classes=num_classes, dropout=0.0)
    logits = head(F_fuse)
    assert logits.shape == (B, num_classes)
    assert torch.isfinite(logits).all()
    LOG.info("  head: Stage3Head OK")


# ---------- Stage3 wrapper ----------
def test_stage3_wrapper():
    """Stage3(T_out) -> logits (B, num_classes); full pipeline."""
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
    LOG.info("  wrapper: Stage3 forward OK")


# ---------- Gradient flow ----------
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
    assert torch.isfinite(T_out.grad).all(), "gradient contains NaN or inf"
    LOG.info("  stage3: gradient flow OK")


# ---------- Determinism ----------
def test_stage3_determinism():
    """Stage3 forward is deterministic for same input (eval mode)."""
    torch.manual_seed(42)
    B, L, D = 2, 12, 32
    num_classes = 5
    T_out = torch.randn(B, 3 * L, D)
    stage3 = Stage3(embed_dim=D, num_classes=num_classes, tokens_per_scale=L, dropout=0.0)
    stage3.eval()
    with torch.no_grad():
        logits1 = stage3(T_out)
        logits2 = stage3(T_out)
    assert torch.allclose(logits1, logits2), "Stage3 forward not deterministic"
    LOG.info("  stage3: determinism OK")


# ---------- Run all ----------
def run_all(verbose: bool = True):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    torch.manual_seed(42)
    LOG.info("========== Stage 3 exhaustive tests ==========")
    test_split_scales()
    test_infer_L()
    test_dual_pool_tokens()
    test_weighted_fusion_shapes()
    test_weighted_fusion_weights_softmax()
    test_stage3_head()
    test_stage3_wrapper()
    test_stage3_gradient_flow()
    test_stage3_determinism()
    LOG.info("========== Stage 3 exhaustive tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 3 exhaustive tests")
    p.add_argument("--quiet", action="store_true", help="Less log output")
    args = p.parse_args()
    run_all(verbose=not args.quiet)
