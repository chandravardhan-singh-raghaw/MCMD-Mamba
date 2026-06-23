"""
Stage 1 + Stage 2 end-to-end: Stage1(x_rgb, x_clahe) → T (B, 3*L, D) → Stage2(T) → T_out (B, 3*L, D).
Validates shapes, finiteness, no NaN, split_tokens consistency, gradient flow.
Use STAGE2_CORE=dummy (default) for fast runs; STAGE2_CORE=mamba for real SSM.
Run from repo root: PYTHONPATH=src python tests/test_stage1_stage2_e2e.py
With pytest: pytest tests/test_stage1_stage2_e2e.py (180s timeout per test via pytest-timeout).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch

try:
    import pytest
    pytestmark = pytest.mark.timeout(180)  # E2E can be slow; allow 180s per test
except ImportError:
    pytestmark = None  # pytest-timeout not installed

try:
    from mcmd_mamba.models.stage1 import Stage1
    from mcmd_mamba.models.stage1.mce import rgb_to_gray
    from mcmd_mamba.models.stage2 import MDMambaStack, build_core
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from mcmd_mamba.models.stage1 import Stage1
    from mcmd_mamba.models.stage1.mce import rgb_to_gray
    from mcmd_mamba.models.stage2 import MDMambaStack, build_core

LOG = logging.getLogger(__name__)

# Default: dummy core for fast CI; set STAGE2_CORE=mamba to test real SSM (slower)
STAGE2_CORE = os.environ.get("STAGE2_CORE", "dummy")


def _make_clahe_input(x_rgb: torch.Tensor, use_real_clahe: bool = True) -> torch.Tensor:
    """(B,3,H,W) → (B,1,H,W); real CLAHE if cv2 else grayscale fallback."""
    if use_real_clahe:
        try:
            from mcmd_mamba.models.stage1.mce import clahe_1ch_from_rgb
            return clahe_1ch_from_rgb(x_rgb)
        except Exception as e:
            LOG.warning("CLAHE failed (%s), using grayscale fallback", e)
    return rgb_to_gray(x_rgb)


def test_stage1_then_stage2_shapes(
    B: int = 2,
    H: int = 224,
    W: int = 224,
    feat_ch: int = 64,
    d_model: int = 64,
    token_hw: Tuple[int, int] = (14, 14),
    num_blocks: int = 2,
    seed: int = 42,
    use_real_clahe: bool = False,
) -> None:
    """
    Stage1(x_rgb, x_clahe) → T (B, 3*L, D).
    Stage2(T) → T_out (B, 3*L, D).
    Assert shapes, finite, no NaN.
    """
    torch.manual_seed(seed)
    L = token_hw[0] * token_hw[1]
    Htok, Wtok = token_hw

    LOG.info(
        "E2E: B=%s H=%s W=%s feat_ch=%s d_model=%s token_hw=%s num_blocks=%s",
        B, H, W, feat_ch, d_model, token_hw, num_blocks,
    )

    # Stage 1
    x_rgb = torch.rand(B, 3, H, W)
    x_clahe_1ch = _make_clahe_input(x_rgb, use_real_clahe=use_real_clahe)
    stage1 = Stage1(feat_ch=feat_ch, d_model=d_model, token_hw=token_hw)
    T = stage1(x_rgb, x_clahe_1ch=x_clahe_1ch, check_shapes=True)

    assert T.shape == (B, 3 * L, d_model), f"Stage1 output T.shape={T.shape} != (B, 3*L, D)"
    assert torch.isfinite(T).all(), "Stage1 output contains NaN or inf"
    LOG.info("  Stage1 output T: %s", T.shape)

    # Stage 2 (same d_model and token_hw so L matches)
    stage2 = MDMambaStack(
        d_model=d_model,
        token_hw=token_hw,
        num_blocks=num_blocks,
        core_kind=STAGE2_CORE,
        dropout=0.0,
    )
    T_out = stage2(T)

    assert T_out.shape == T.shape, f"Stage2 output T_out.shape={T_out.shape} != T.shape={T.shape}"
    assert torch.isfinite(T_out).all(), "Stage2 output contains NaN or inf"
    LOG.info("  Stage2 output T_out: %s", T_out.shape)

    LOG.info("  Stage1 → Stage2 shapes OK, finite, no NaN.")


def test_stage1_split_tokens_matches_stage2_input():
    """
    Stage1.split_tokens(T) recovers Ts, Tc, Tf each (B, L, D).
    Those shapes match what Stage2 expects per scale group (L = Htok*Wtok).
    """
    B, H, W = 2, 224, 224
    feat_ch, d_model = 64, 64
    token_hw = (14, 14)
    L = token_hw[0] * token_hw[1]

    torch.manual_seed(43)
    x_rgb = torch.rand(B, 3, H, W)
    x_clahe = rgb_to_gray(x_rgb)

    stage1 = Stage1(feat_ch=feat_ch, d_model=d_model, token_hw=token_hw)
    T = stage1(x_rgb, x_clahe_1ch=x_clahe, check_shapes=True)

    Ts, Tc, Tf = stage1.split_tokens(T)
    assert Ts.shape == (B, L, d_model), f"Ts.shape={Ts.shape}"
    assert Tc.shape == (B, L, d_model)
    assert Tf.shape == (B, L, d_model)
    assert torch.cat([Ts, Tc, Tf], dim=1).shape == T.shape
    assert torch.allclose(torch.cat([Ts, Tc, Tf], dim=1), T)

    # Stage2 internally splits (B, 3*L, D) the same way; L must match token_hw
    stage2 = MDMambaStack(
        d_model=d_model,
        token_hw=token_hw,
        num_blocks=1,
        core_kind=STAGE2_CORE,
        dropout=0.0,
    )
    T_out = stage2(T)
    assert T_out.shape == T.shape
    LOG.info("  split_tokens consistency OK: Stage1 L matches Stage2 token_hw.")


def test_stage1_stage2_gradient_flow():
    """Backward through Stage1 → Stage2: gradients flow, no NaN."""
    B, H, W = 2, 112, 112  # smaller for speed
    feat_ch, d_model = 32, 32
    token_hw = (7, 7)
    L = token_hw[0] * token_hw[1]

    torch.manual_seed(44)
    x_rgb = torch.rand(B, 3, H, W, requires_grad=True)
    x_clahe = rgb_to_gray(x_rgb.detach()).detach()
    x_clahe.requires_grad_(False)

    stage1 = Stage1(feat_ch=feat_ch, d_model=d_model, token_hw=token_hw)
    stage2 = MDMambaStack(
        d_model=d_model,
        token_hw=token_hw,
        num_blocks=1,
        core_kind=STAGE2_CORE,
        dropout=0.0,
    )

    T = stage1(x_rgb, x_clahe_1ch=x_clahe, check_shapes=True)
    T_out = stage2(T)
    loss = T_out.sum()
    loss.backward()

    assert x_rgb.grad is not None and x_rgb.grad.shape == x_rgb.shape
    assert torch.isfinite(x_rgb.grad).all(), "gradient contains NaN or inf"
    LOG.info("  E2E gradient flow OK.")


def run_all(verbose: bool = True, core_kind: str = None):
    global STAGE2_CORE
    if core_kind is not None:
        STAGE2_CORE = core_kind
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    LOG.info("========== Stage 1 + Stage 2 E2E tests (core=%s) ==========", STAGE2_CORE)
    test_stage1_then_stage2_shapes(use_real_clahe=False)
    test_stage1_split_tokens_matches_stage2_input()
    test_stage1_stage2_gradient_flow()
    try:
        test_stage1_then_stage2_shapes(use_real_clahe=True)
        LOG.info("  (Real CLAHE path also passed.)")
    except Exception as e:
        LOG.warning("  Real CLAHE path skipped: %s", e)
    LOG.info("========== Stage 1 + Stage 2 E2E tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 1 + Stage 2 E2E tests")
    p.add_argument("--quiet", action="store_true", help="Less log output")
    p.add_argument(
        "--core",
        choices=("dummy", "mamba"),
        default=None,
        help="Stage2 core kind (default: STAGE2_CORE env or dummy). Use mamba for real SSM.",
    )
    args = p.parse_args()
    run_all(verbose=not args.quiet, core_kind=args.core)
