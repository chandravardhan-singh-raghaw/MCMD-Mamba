"""
Exhaustive Stage 2 tests: full MD-Mamba architecture (scans, ops, mamba_core, md_ssm, block, stack).
Validates shapes, roundtrips, finiteness, gradient flow. Use --core mamba to test real SSM (slower).
Run from repo root: PYTHONPATH=src python tests/test_stage2_exhaustive.py
With pytest: pytest tests/test_stage2_exhaustive.py -v
"""

import logging
import os
import sys
from pathlib import Path

import torch

try:
    from mcmd_mamba.models.stage2 import (
        horizontal_indices,
        vertical_indices,
        spiral_indices_topright_ccw,
        apply_index,
        invert_index,
        SequenceCore,
        DummyCore,
        MambaSSMCore,
        build_core,
        assert_core_io,
        MDSSM,
        MDMambaBlock,
        MDMambaStack,
    )
    from mcmd_mamba.models.stage2.md_ssm import (
        split_concat_sequence,
        merge_concat_sequence,
        compute_p_branch,
        compute_g_branch,
        run_scan_path,
        merge_paths,
        gated_output,
        assert_mdssm_io,
    )
    from mcmd_mamba.models.stage2.md_mamba_block import FFN, residual_add, assert_block_io
    from mcmd_mamba.models.stage2.ops import split_scales, merge_scales, scan_process_unscan
except ImportError:
    _src = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(_src))
    from mcmd_mamba.models.stage2 import (
        horizontal_indices,
        vertical_indices,
        spiral_indices_topright_ccw,
        apply_index,
        invert_index,
        SequenceCore,
        DummyCore,
        MambaSSMCore,
        build_core,
        assert_core_io,
        MDSSM,
        MDMambaBlock,
        MDMambaStack,
    )
    from mcmd_mamba.models.stage2.md_ssm import (
        split_concat_sequence,
        merge_concat_sequence,
        compute_p_branch,
        compute_g_branch,
        run_scan_path,
        merge_paths,
        gated_output,
        assert_mdssm_io,
    )
    from mcmd_mamba.models.stage2.md_mamba_block import FFN, residual_add, assert_block_io
    from mcmd_mamba.models.stage2.ops import split_scales, merge_scales, scan_process_unscan

LOG = logging.getLogger(__name__)

# Default: dummy core for fast CI; set STAGE2_CORE=mamba to test real SSM
STAGE2_CORE = os.environ.get("STAGE2_CORE", "dummy")


# ---------- Scans ----------
def test_scans_all_roundtrips():
    """All three scan orders roundtrip (apply_index then invert_index recovers input)."""
    H, W = 4, 5
    B, L, D = 2, H * W, 8
    x = torch.randn(B, L, D)
    for name, idx_fn in [
        ("horizontal", horizontal_indices),
        ("vertical", vertical_indices),
        ("spiral_topright_ccw", spiral_indices_topright_ccw),
    ]:
        idx = idx_fn(H, W)
        inv = invert_index(idx)
        assert idx.shape == (L,), f"{name}: idx.shape={idx.shape}"
        y = apply_index(x, idx, dim=1)
        z = apply_index(y, inv, dim=1)
        assert torch.allclose(x, z), f"{name} roundtrip failed"
    LOG.info("  scans: all roundtrips OK")


def test_scans_3x3_spec():
    """3×3 grid: horizontal [0..8], vertical [0,3,6,1,4,7,2,5,8], spiral [2,5,8,7,6,3,0,1,4]."""
    H, W = 3, 3
    idx_h = horizontal_indices(H, W)
    idx_v = vertical_indices(H, W)
    idx_s = spiral_indices_topright_ccw(H, W)
    assert list(idx_h.tolist()) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert list(idx_v.tolist()) == [0, 3, 6, 1, 4, 7, 2, 5, 8]
    assert list(idx_s.tolist()) == [2, 5, 8, 7, 6, 3, 0, 1, 4]
    LOG.info("  scans: 3×3 spec OK")


# ---------- Ops ----------
def test_ops_split_merge_roundtrip():
    """split_scales then merge_scales recovers input."""
    B, L, D = 2, 6, 8
    T = torch.randn(B, 3 * L, D)
    Ts, Tc, Tf = split_scales(T, L)
    T2 = merge_scales(Ts, Tc, Tf)
    assert torch.allclose(T, T2)
    LOG.info("  ops: split_scales / merge_scales roundtrip OK")


def test_ops_scan_process_unscan_roundtrip():
    """scan_process_unscan with identity core recovers input."""
    B, L, D = 2, 12, 8
    x = torch.randn(B, L, D)
    idx = horizontal_indices(3, 4)
    y = scan_process_unscan(x, idx, lambda z: z)
    assert torch.allclose(x, y)
    LOG.info("  ops: scan_process_unscan (identity) roundtrip OK")


# ---------- Mamba core ----------
def test_core_dummy_forward():
    """DummyCore: (B, L, D) → (B, L, D), finite, assert_core_io passes."""
    B, L, D = 2, 10, 8
    x = torch.randn(B, L, D)
    core = build_core("dummy", D, dropout=0.0)
    y = core(x)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert torch.isfinite(y).all()
    assert_core_io(x, y, D)
    LOG.info("  mamba_core: DummyCore forward OK")


def test_core_mamba_forward():
    """MambaSSMCore (real SSM): (B, L, D) → (B, L, D), finite."""
    B, L, D = 2, 10, 8
    x = torch.randn(B, L, D)
    core = build_core("mamba", D)
    y = core(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert_core_io(x, y, D)
    LOG.info("  mamba_core: MambaSSMCore forward OK")


def test_core_build_factory():
    """build_core('dummy'/'mamba') returns correct type; unknown kind raises."""
    core_d = build_core("dummy", 8)
    core_m = build_core("mamba", 8)
    assert isinstance(core_d, DummyCore)
    assert isinstance(core_m, MambaSSMCore)
    try:
        build_core("unknown", 8)
        assert False, "build_core('unknown') should raise"
    except ValueError:
        pass
    LOG.info("  mamba_core: build_core factory OK")


# ---------- MD-SSM ----------
def test_mdssm_helpers():
    """split_concat_sequence, merge_concat_sequence, p/g branches, merge_paths, gated_output shapes."""
    B, L, D = 2, 6, 8
    H, W = 2, 3
    x = torch.randn(B, 3 * L, D)
    xs, xc, xf = split_concat_sequence(x, L)
    assert xs.shape == (B, L, D) and xc.shape == (B, L, D) and xf.shape == (B, L, D)
    y = merge_concat_sequence(xs, xc, xf)
    assert y.shape == x.shape and torch.allclose(x, y)

    p_linear = torch.nn.Linear(D, D)
    p_conv1d = torch.nn.Conv1d(D, D, 3, padding=1)
    g_linear = torch.nn.Linear(D, D)
    merge_ln = torch.nn.LayerNorm(D)
    out_linear = torch.nn.Linear(D, D)

    p = compute_p_branch(xs, p_linear, p_conv1d)
    g = compute_g_branch(xs, g_linear)
    assert p.shape == (B, L, D) and g.shape == (B, L, D)

    core = build_core("dummy", D)
    idx_h = horizontal_indices(H, W)
    px = run_scan_path(p, idx_h, core)
    assert px.shape == (B, L, D)

    sigma = merge_paths(px, px, px, merge_ln, merge="sum")
    assert sigma.shape == (B, L, D)
    out = gated_output(sigma, g, out_linear)
    assert out.shape == (B, L, D)
    LOG.info("  md_ssm: all helpers OK")


def test_mdssm_full_forward():
    """MDSSM: (B, 3L, D) → (B, 3L, D), assert_mdssm_io, finite."""
    B, L, D = 2, 12, 16
    H, W = 3, 4
    x = torch.randn(B, 3 * L, D)
    core = build_core(STAGE2_CORE, D, dropout=0.0)
    m = MDSSM(d_model=D, token_hw=(H, W), core=core, conv_kernel=3, merge="sum")
    y = m(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert_mdssm_io(x, y, L, D)
    LOG.info("  md_ssm: MDSSM full forward OK")


# ---------- MD-Mamba block ----------
def test_block_ffn():
    """FFN: (B, N, D) → (B, N, D)."""
    B, N, D = 2, 36, 16
    x = torch.randn(B, N, D)
    ffn = FFN(d_model=D, mult=4, dropout=0.0)
    y = ffn(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    LOG.info("  md_mamba_block: FFN OK")


def test_block_residual_add():
    """residual_add(x, delta) == x + delta, shape assert."""
    B, N, D = 2, 36, 16
    x = torch.randn(B, N, D)
    delta = torch.randn(B, N, D)
    out = residual_add(x, delta)
    assert torch.allclose(out, x + delta)
    LOG.info("  md_mamba_block: residual_add OK")


def test_block_mdmamba_forward():
    """MDMambaBlock: (B, 3L, D) → (B, 3L, D), assert_block_io."""
    B, L, D = 2, 12, 16
    H, W = 3, 4
    x = torch.randn(B, 3 * L, D)
    core = build_core(STAGE2_CORE, D, dropout=0.0)
    md_ssm = MDSSM(d_model=D, token_hw=(H, W), core=core, conv_kernel=3, merge="sum")
    block = MDMambaBlock(d_model=D, md_ssm=md_ssm, dropout=0.0)
    y = block(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert_block_io(x, y, D)
    LOG.info("  md_mamba_block: MDMambaBlock forward OK")


# ---------- Stack ----------
def test_stack_forward():
    """MDMambaStack: (B, 3L, D) → (B, 3L, D), K blocks. Uses STAGE2_CORE (dummy/mamba)."""
    B, L, D = 2, 12, 16
    H, W = 3, 4
    x = torch.randn(B, 3 * L, D)
    stack = MDMambaStack(
        d_model=D, token_hw=(H, W), num_blocks=3, core_kind=STAGE2_CORE, dropout=0.0
    )
    y = stack(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    LOG.info("  stack: MDMambaStack forward OK (core=%s)", STAGE2_CORE)


# ---------- Gradient flow ----------
def test_stage2_gradient_flow():
    """Full stack backward: gradients flow, no NaN."""
    B, L, D = 2, 12, 16
    H, W = 3, 4
    x = torch.randn(B, 3 * L, D, requires_grad=True)
    stack = MDMambaStack(
        d_model=D, token_hw=(H, W), num_blocks=2, core_kind=STAGE2_CORE, dropout=0.0
    )
    y = stack(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all(), "gradient contains NaN or inf"
    LOG.info("  stage2: gradient flow OK")


# ---------- Run all ----------
def run_all(verbose: bool = True, core_kind: str = None):
    global STAGE2_CORE
    if core_kind is not None:
        STAGE2_CORE = core_kind
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    torch.manual_seed(42)
    LOG.info("========== Stage 2 exhaustive tests (core=%s) ==========", STAGE2_CORE)
    test_scans_all_roundtrips()
    test_scans_3x3_spec()
    test_ops_split_merge_roundtrip()
    test_ops_scan_process_unscan_roundtrip()
    test_core_dummy_forward()
    test_core_mamba_forward()
    test_core_build_factory()
    test_mdssm_helpers()
    test_mdssm_full_forward()
    test_block_ffn()
    test_block_residual_add()
    test_block_mdmamba_forward()
    test_stack_forward()
    test_stage2_gradient_flow()
    LOG.info("========== Stage 2 exhaustive tests passed ==========")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 2 exhaustive tests")
    p.add_argument("--quiet", action="store_true", help="Less log output")
    p.add_argument(
        "--core",
        choices=("dummy", "mamba"),
        default=None,
        help="Core kind (default: STAGE2_CORE env or dummy). Use mamba for real SSM.",
    )
    args = p.parse_args()
    run_all(verbose=not args.quiet, core_kind=args.core)
