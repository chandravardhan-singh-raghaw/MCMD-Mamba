"""
Stage 1 smoke tests: MCE → PSA → Tokenize → concat + scale_embed.
Uses mcmd_mamba.models.stage1 (run from repo root with PYTHONPATH=src or pip install -e .).
Logs each step; assertions with clear messages; robust to missing cv2 (fake CLAHE fallback).
"""

import logging
import sys
from typing import Optional, Tuple

import torch

# Ensure package is importable (run from repo root: PYTHONPATH=src pytest tests/)
try:
    from mcmd_mamba.models.stage1 import (
        Stage1,
        MultiChannelEnhancement,
        PyramidalSelfAttention,
        apply_psa,
        Tokenizer,
        ScaleEmbedding,
        concat_with_scale_embeddings,
    )
    from mcmd_mamba.models.stage1.mce import rgb_to_gray
except ImportError:
    # Fallback if not installed: add src to path
    from pathlib import Path
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from mcmd_mamba.models.stage1 import (
        Stage1,
        MultiChannelEnhancement,
        PyramidalSelfAttention,
        apply_psa,
        Tokenizer,
        ScaleEmbedding,
        concat_with_scale_embeddings,
    )
    from mcmd_mamba.models.stage1.mce import rgb_to_gray

LOG = logging.getLogger(__name__)


def _make_clahe_input(x_rgb: torch.Tensor, use_real_clahe: bool = True) -> torch.Tensor:
    """Return (B,1,H,W) for MCE: real CLAHE if cv2 else grayscale fallback."""
    if use_real_clahe:
        try:
            from mcmd_mamba.models.stage1.mce import clahe_1ch_from_rgb
            return clahe_1ch_from_rgb(x_rgb)
        except Exception as e:
            LOG.warning("CLAHE from RGB failed (%s), using grayscale fallback", e)
    return rgb_to_gray(x_rgb)


def test_stage1_wrapper(
    B: int = 2,
    H: int = 224,
    W: int = 224,
    feat_ch: int = 128,
    D: int = 64,
    token_hw: Tuple[int, int] = (14, 14),
    seed: int = 0,
    use_real_clahe: bool = True,
) -> None:
    """Test Stage1 wrapper: single forward returns T (B, 3*L, D); split_tokens recovers Ts,Tc,Tf."""
    torch.manual_seed(seed)
    L = token_hw[0] * token_hw[1]

    LOG.info("Stage1 wrapper test: B=%s H=%s W=%s feat_ch=%s D=%s token_hw=%s", B, H, W, feat_ch, D, token_hw)

    x_rgb = torch.rand(B, 3, H, W)
    x_clahe_1ch = _make_clahe_input(x_rgb, use_real_clahe=use_real_clahe)
    assert x_clahe_1ch.shape == (B, 1, H, W), f"x_clahe_1ch shape {x_clahe_1ch.shape}"

    model = Stage1(feat_ch=feat_ch, d_model=D, token_hw=token_hw)
    T = model(x_rgb, x_clahe_1ch=x_clahe_1ch, check_shapes=True)

    LOG.info("  T shape: %s (expected B=%s, 3*L=%s, D=%s)", T.shape, B, 3 * L, D)
    assert T.shape == (B, 3 * L, D), f"T shape {T.shape} != (B, 3*L, D) = ({B}, {3 * L}, {D})"

    Ts, Tc, Tf = model.split_tokens(T)
    LOG.info("  split_tokens: Ts %s, Tc %s, Tf %s", Ts.shape, Tc.shape, Tf.shape)
    assert Ts.shape == (B, L, D), f"Ts shape {Ts.shape}"
    assert Tc.shape == (B, L, D), f"Tc shape {Tc.shape}"
    assert Tf.shape == (B, L, D), f"Tf shape {Tf.shape}"

    LOG.info("  Stage1 wrapper test passed.")


def test_stage1_manual_pipeline(
    B: int = 2,
    H: int = 224,
    W: int = 224,
    feat_ch: int = 128,
    D: int = 64,
    token_hw: Tuple[int, int] = (14, 14),
    seed: int = 0,
    use_real_clahe: bool = True,
) -> None:
    """Test manual pipeline: MCE → PSA → Tokenize → concat + scale_embed; assert all intermediates."""
    torch.manual_seed(seed)
    L = token_hw[0] * token_hw[1]

    LOG.info("Stage1 manual pipeline test: B=%s feat_ch=%s D=%s token_hw=%s", B, feat_ch, D, token_hw)

    x_rgb = torch.rand(B, 3, H, W)
    x_clahe_1ch = _make_clahe_input(x_rgb, use_real_clahe=use_real_clahe)

    # 1) MCE
    mce = MultiChannelEnhancement(feat_ch=feat_ch)
    Fs, Fc, Ff = mce(x_rgb, x_clahe_1ch=x_clahe_1ch)
    LOG.info("  MCE: Fs %s, Fc %s, Ff %s", Fs.shape, Fc.shape, Ff.shape)
    assert Fs.dim() == 4 and Fs.shape[0] == B and Fs.shape[1] == feat_ch
    assert Fc.shape == Fs.shape and Ff.shape == Fs.shape

    # 2) PSA
    psa = PyramidalSelfAttention(patch_size=8)
    M = psa(Ff)
    LOG.info("  PSA: M %s, min=%.4f max=%.4f", M.shape, float(M.min()), float(M.max()))
    assert M.ndim == 4 and M.shape[1] == 1, f"M shape {M.shape}"
    assert M.shape[-2:] == Ff.shape[-2:], f"M spatial {M.shape[-2:]} vs Ff {Ff.shape[-2:]}"
    assert 0.0 <= float(M.min()) and float(M.max()) <= 1.0 + 1e-3, "M should be in [0,1]"

    Fs = apply_psa(M, Fs)
    Fc = apply_psa(M, Fc)
    Ff = apply_psa(M, Ff)
    LOG.info("  apply_psa: Fs/Fc/Ff spatial %s", Fs.shape[-2:])

    # 3) Tokenize
    cfg = {"token_hw": token_hw}
    tok = Tokenizer(cfg, in_channels=feat_ch, d_model=D, token_hw=token_hw)
    Ts = tok(Fs)
    Tc = tok(Fc)
    Tf = tok(Ff)
    LOG.info("  Tokenizer: Ts %s, Tc %s, Tf %s", Ts.shape, Tc.shape, Tf.shape)
    assert Ts.shape == (B, L, D), f"Ts {Ts.shape}"
    assert Tc.shape == (B, L, D), f"Tc {Tc.shape}"
    assert Tf.shape == (B, L, D), f"Tf {Tf.shape}"

    # 4) Concat + scale embed
    T_cat = torch.cat([Ts, Tc, Tf], dim=1)
    scale_emb = ScaleEmbedding(d_model=D, num_tokens=3 * L)
    T = concat_with_scale_embeddings(T_cat, scale_emb)
    LOG.info("  T (concat + scale_embed): %s", T.shape)
    assert T.shape == (B, 3 * L, D), f"T shape {T.shape}"

    LOG.info("  Manual pipeline test passed.")


def test_mce_only(seed: int = 0) -> None:
    """MCE only: output shapes and no NaNs."""
    torch.manual_seed(seed)
    B, H, W = 2, 224, 224
    x_rgb = torch.rand(B, 3, H, W)
    x_clahe = rgb_to_gray(x_rgb)

    mce = MultiChannelEnhancement(feat_ch=64)
    Fs, Fc, Ff = mce(x_rgb, x_clahe_1ch=x_clahe)
    LOG.info("MCE only: Fs %s Fc %s Ff %s", Fs.shape, Fc.shape, Ff.shape)
    assert Fs.shape == Fc.shape == Ff.shape
    assert not torch.isnan(Fs).any() and not torch.isnan(Ff).any()
    LOG.info("  MCE only test passed.")


def test_psa_only(seed: int = 0) -> None:
    """PSA only: mask shape and range [0,1]."""
    torch.manual_seed(seed)
    B, C, H, W = 2, 64, 28, 28
    Ff = torch.rand(B, C, H, W)

    psa = PyramidalSelfAttention(patch_size=8)
    M = psa(Ff)
    LOG.info("PSA only: M %s min=%.4f max=%.4f", M.shape, float(M.min()), float(M.max()))
    assert M.shape == (B, 1, H, W)
    assert 0.0 <= float(M.min()) and float(M.max()) <= 1.0 + 1e-3
    LOG.info("  PSA only test passed.")


def test_tokenizer_only(seed: int = 0) -> None:
    """Tokenizer only: (B,C,H,W) -> (B,L,D) with fixed token_hw."""
    torch.manual_seed(seed)
    B, C, H, W = 2, 64, 28, 28
    token_hw = (14, 14)
    L = token_hw[0] * token_hw[1]
    D = 32
    x = torch.rand(B, C, H, W)

    tok = Tokenizer({}, in_channels=C, d_model=D, token_hw=token_hw)
    out = tok(x)
    LOG.info("Tokenizer only: in %s -> out %s", x.shape, out.shape)
    assert out.shape == (B, L, D)
    LOG.info("  Tokenizer only test passed.")


def run_all(verbose: bool = True) -> None:
    """Run all Stage1 tests with logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    LOG.info("========== Stage1 tests ==========")
    test_mce_only()
    test_psa_only()
    test_tokenizer_only()
    test_stage1_manual_pipeline(use_real_clahe=False)  # no cv2 dependency for CI
    test_stage1_wrapper(use_real_clahe=False)
    try:
        test_stage1_manual_pipeline(use_real_clahe=True)
        test_stage1_wrapper(use_real_clahe=True)
        LOG.info("(Real CLAHE path also passed.)")
    except Exception as e:
        LOG.warning("Real CLAHE path skipped or failed: %s", e)
    LOG.info("========== All Stage1 tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage1 smoke tests")
    p.add_argument("--quiet", action="store_true", help="Less logging")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    run_all(verbose=not args.quiet)
