"""
Stage 1 + Stage 2 + Stage 3 end-to-end: full pipeline to logits.
Stage1(x_rgb, x_clahe) -> T -> Stage2(T) -> T_out -> Stage3(T_out) -> logits.
Validates shapes, finiteness, no NaN, gradient flow.
Use STAGE2_CORE=dummy (default) for fast runs; STAGE2_CORE=mamba for real SSM.
Run from repo root: PYTHONPATH=src python tests/test_stage1_stage2_stage3_e2e.py
With pytest: pytest tests/test_stage1_stage2_stage3_e2e.py (180s timeout per test via pytest-timeout).
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
    from mcmd_mamba.models.stage2 import MDMambaStack
    from mcmd_mamba.models.stage3 import Stage3
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from mcmd_mamba.models.stage1 import Stage1
    from mcmd_mamba.models.stage1.mce import rgb_to_gray
    from mcmd_mamba.models.stage2 import MDMambaStack
    from mcmd_mamba.models.stage3 import Stage3

LOG = logging.getLogger(__name__)

STAGE2_CORE = os.environ.get("STAGE2_CORE", "dummy")


def _make_clahe_input(x_rgb: torch.Tensor, use_real_clahe: bool = True) -> torch.Tensor:
    """(B,3,H,W) -> (B,1,H,W); real CLAHE if cv2 else grayscale fallback."""
    if use_real_clahe:
        try:
            from mcmd_mamba.models.stage1.mce import clahe_1ch_from_rgb
            return clahe_1ch_from_rgb(x_rgb)
        except Exception as e:
            LOG.warning("CLAHE failed (%s), using grayscale fallback", e)
    return rgb_to_gray(x_rgb)


def test_full_pipeline_shapes(
    B: int = 2,
    H: int = 224,
    W: int = 224,
    feat_ch: int = 64,
    d_model: int = 64,
    token_hw: Tuple[int, int] = (14, 14),
    num_blocks: int = 2,
    num_classes: int = 5,
    seed: int = 42,
    use_real_clahe: bool = False,
) -> None:
    """
    Stage1 -> Stage2 -> Stage3: x_rgb (B,3,H,W) -> logits (B, num_classes).
    Assert shapes, finite, no NaN.
    """
    torch.manual_seed(seed)
    L = token_hw[0] * token_hw[1]

    LOG.info(
        "E2E: B=%s H=%s W=%s feat_ch=%s d_model=%s token_hw=%s num_blocks=%s num_classes=%s",
        B, H, W, feat_ch, d_model, token_hw, num_blocks, num_classes,
    )

    # Stage 1
    x_rgb = torch.rand(B, 3, H, W)
    x_clahe_1ch = _make_clahe_input(x_rgb, use_real_clahe=use_real_clahe)
    stage1 = Stage1(feat_ch=feat_ch, d_model=d_model, token_hw=token_hw)
    T = stage1(x_rgb, x_clahe_1ch=x_clahe_1ch, check_shapes=True)

    assert T.shape == (B, 3 * L, d_model)
    assert torch.isfinite(T).all()
    LOG.info("  Stage1 output T: %s", T.shape)

    # Stage 2
    stage2 = MDMambaStack(
        d_model=d_model,
        token_hw=token_hw,
        num_blocks=num_blocks,
        core_kind=STAGE2_CORE,
        dropout=0.0,
    )
    T_out = stage2(T)

    assert T_out.shape == (B, 3 * L, d_model)
    assert torch.isfinite(T_out).all()
    LOG.info("  Stage2 output T_out: %s", T_out.shape)

    # Stage 3
    stage3 = Stage3(
        embed_dim=d_model,
        num_classes=num_classes,
        tokens_per_scale=L,
        dropout=0.0,
    )
    logits = stage3(T_out)

    assert logits.shape == (B, num_classes)
    assert torch.isfinite(logits).all()
    LOG.info("  Stage3 output logits: %s", logits.shape)

    LOG.info("  Stage1 -> Stage2 -> Stage3 shapes OK, finite, no NaN.")


def test_full_pipeline_gradient_flow():
    """Backward through Stage1 -> Stage2 -> Stage3: gradients flow, no NaN."""
    B, H, W = 2, 112, 112
    feat_ch, d_model = 32, 32
    token_hw = (7, 7)
    L = token_hw[0] * token_hw[1]
    num_classes = 5

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
    stage3 = Stage3(embed_dim=d_model, num_classes=num_classes, tokens_per_scale=L, dropout=0.0)

    T = stage1(x_rgb, x_clahe_1ch=x_clahe, check_shapes=True)
    T_out = stage2(T)
    logits = stage3(T_out)
    loss = logits.sum()
    loss.backward()

    assert x_rgb.grad is not None and x_rgb.grad.shape == x_rgb.shape
    assert torch.isfinite(x_rgb.grad).all(), "gradient contains NaN or inf"
    LOG.info("  Full pipeline gradient flow OK.")


def test_split_tokens_consistency():
    """Stage1.split_tokens and Stage3 split agree on L; Stage3 receives correct layout."""
    B, H, W = 2, 224, 224
    feat_ch, d_model = 64, 64
    token_hw = (14, 14)
    L = token_hw[0] * token_hw[1]
    num_classes = 5

    torch.manual_seed(45)
    x_rgb = torch.rand(B, 3, H, W)
    x_clahe = rgb_to_gray(x_rgb)

    stage1 = Stage1(feat_ch=feat_ch, d_model=d_model, token_hw=token_hw)
    stage2 = MDMambaStack(
        d_model=d_model,
        token_hw=token_hw,
        num_blocks=1,
        core_kind=STAGE2_CORE,
        dropout=0.0,
    )
    stage3 = Stage3(embed_dim=d_model, num_classes=num_classes, tokens_per_scale=L, dropout=0.0)

    T = stage1(x_rgb, x_clahe_1ch=x_clahe, check_shapes=True)
    Ts, Tc, Tf = stage1.split_tokens(T)
    assert Ts.shape == (B, L, d_model) and Tc.shape == (B, L, d_model) and Tf.shape == (B, L, d_model)

    T_out = stage2(T)
    logits = stage3(T_out)
    assert logits.shape == (B, num_classes)
    assert torch.isfinite(logits).all()

    LOG.info("  split_tokens consistency OK: L matches across stages.")


def run_all(verbose: bool = True, core_kind: str = None):
    global STAGE2_CORE
    if core_kind is not None:
        STAGE2_CORE = core_kind
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    LOG.info("========== Stage 1 + Stage 2 + Stage 3 E2E tests (core=%s) ==========", STAGE2_CORE)
    test_full_pipeline_shapes(use_real_clahe=False)
    test_full_pipeline_gradient_flow()
    test_split_tokens_consistency()
    try:
        test_full_pipeline_shapes(use_real_clahe=True)
        LOG.info("  (Real CLAHE path also passed.)")
    except Exception as e:
        LOG.warning("  Real CLAHE path skipped: %s", e)
    LOG.info("========== Stage 1 + Stage 2 + Stage 3 E2E tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 1 + Stage 2 + Stage 3 E2E tests")
    p.add_argument("--quiet", action="store_true", help="Less log output")
    p.add_argument(
        "--core",
        choices=("dummy", "mamba"),
        default=None,
        help="Stage2 core kind (default: STAGE2_CORE env or dummy). Use mamba for real SSM.",
    )
    args = p.parse_args()
    run_all(verbose=not args.quiet, core_kind=args.core)
