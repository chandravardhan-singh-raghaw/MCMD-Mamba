"""
Eq. (8) Multi-Direction State Space Model: p branch + scan paths + merge + gated output.
Scan index generation stays in scans.py. Full API/equation guide: see doc.md in this package.

Implements:
  (8a) p = SiLU(Conv1D(Linear(x)))
  (8b) Σ = LayerNorm(Ψx(p) + Ψy(p) + Ψθ(p))
  (8c) g = SiLU(Linear(x))
  (8d) yout = Linear(Σ ⊙ g)
"""

from typing import Tuple

__all__ = [
    "split_concat_sequence",
    "merge_concat_sequence",
    "compute_p_branch",
    "compute_g_branch",
    "run_scan_path",
    "merge_paths",
    "gated_output",
    "assert_mdssm_io",
    "MDSSM",
]

import torch
import torch.nn as nn

from .mamba_core import SequenceCore
from .scans import (
    horizontal_indices,
    vertical_indices,
    spiral_indices_topright_ccw,
)
from .ops import scan_process_unscan


def split_concat_sequence(
    x: torch.Tensor,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split concatenated tokens (doc.md).
    Input (B, 3L, D) → output (xs, xc, xf) each (B, L, D).
    """
    assert L > 0, "split_concat_sequence: L must be positive"
    B, N, D = x.shape
    assert N == 3 * L, f"split_concat_sequence: x.shape[1]={N} != 3*L={3*L}"
    xs = x[:, :L, :]
    xc = x[:, L : 2 * L, :]
    xf = x[:, 2 * L :, :]
    assert xs.shape == (B, L, D) and xc.shape == (B, L, D) and xf.shape == (B, L, D)
    return xs, xc, xf


def merge_concat_sequence(
    xs: torch.Tensor,
    xc: torch.Tensor,
    xf: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate back to (B, 3L, D) (doc.md).
    """
    assert xs.dim() == 3 and xc.dim() == 3 and xf.dim() == 3
    assert xs.shape == xc.shape == xf.shape, (
        f"merge_concat_sequence: shapes must match, got xs={xs.shape} xc={xc.shape} xf={xf.shape}"
    )
    B, L, D = xs.shape
    y = torch.cat([xs, xc, xf], dim=1)
    assert y.shape == (B, 3 * L, D)
    return y


def compute_p_branch(
    xg: torch.Tensor,
    p_linear: nn.Module,
    p_conv1d: nn.Module,
) -> torch.Tensor:
    """
    Eq. (8a) for one scale group (doc.md): p = SiLU(Conv1D(Linear(xg))).
    xg: (B, L, D) → p: (B, L, D).
    """
    B, L, D = xg.shape
    h = p_linear(xg)                    # (B, L, D)
    h = h.transpose(1, 2)               # (B, D, L) for Conv1d
    h = p_conv1d(h)
    h = h.transpose(1, 2)               # (B, L, D)
    p = torch.nn.functional.silu(h)
    assert p.shape == (B, L, D)
    return p


def compute_g_branch(xg: torch.Tensor, g_linear: nn.Module) -> torch.Tensor:
    """
    Eq. (8c) for one scale group (doc.md): g = SiLU(Linear(xg)).
    xg: (B, L, D) → g: (B, L, D).
    """
    g = torch.nn.functional.silu(g_linear(xg))
    assert g.shape == xg.shape
    return g


def run_scan_path(
    p: torch.Tensor,
    idx: torch.Tensor,
    core: SequenceCore,
) -> torch.Tensor:
    """
    One scan operator Ψ (doc.md): reorder by idx → apply core → undo reorder.
    p: (B, L, D), idx: (L,) permutation, core: SequenceCore.
    Returns (B, L, D) aligned to original token positions.
    """
    out = scan_process_unscan(p, idx, core)
    assert out.shape == p.shape, f"run_scan_path: output {out.shape} != input {p.shape}"
    return out


def merge_paths(
    px: torch.Tensor,
    py: torch.Tensor,
    ps: torch.Tensor,
    merge_ln: nn.Module,
    merge: str = "sum",
) -> torch.Tensor:
    """
    Eq. (8b) (doc.md): Σ = LayerNorm(Merge(Ψx(p), Ψy(p), Ψθ(p))).
    Merge is summation: px + py + ps. Returns Sigma (B, L, D).
    """
    assert px.shape == py.shape == ps.shape, (
        f"merge_paths: px={px.shape} py={py.shape} ps={ps.shape} must match"
    )
    if merge != "sum":
        raise ValueError(f"merge_paths: only merge='sum' is implemented, got {merge!r}")
    sigma = merge_ln(px + py + ps)
    assert sigma.shape == px.shape
    return sigma


def gated_output(
    Sigma: torch.Tensor,
    g: torch.Tensor,
    out_linear: nn.Module,
) -> torch.Tensor:
    """
    Eq. (8d) (doc.md): yout = Linear(Σ ⊙ g).
    Sigma, g: (B, L, D) → (B, L, D).
    """
    assert Sigma.shape == g.shape, f"gated_output: Sigma={Sigma.shape} g={g.shape} must match"
    return out_linear(Sigma * g)


def assert_mdssm_io(
    x: torch.Tensor,
    y: torch.Tensor,
    L: int,
    d_model: int,
) -> None:
    """
    Validation helper (doc.md): input/output shapes (B, 3L, D), outputs finite, no NaN/inf.
    """
    assert x.ndim == 3 and y.ndim == 3
    assert x.shape == y.shape, f"input/output shapes must match: x={x.shape} y={y.shape}"
    assert x.shape[1] == 3 * L, f"x.shape[1]={x.shape[1]} != 3*L={3*L}"
    assert x.shape[2] == d_model and y.shape[2] == d_model
    assert torch.isfinite(x).all(), "x contains NaN or inf"
    assert torch.isfinite(y).all(), "y contains NaN or inf"


class MDSSM(nn.Module):
    """
    Eq. (8) Multi-Direction State Space Model (doc.md).

    Computes MD-SSM over (B, 3L, D) by applying Eq. (8a–8d) on each of the three
    scale groups (structural, contextual, fine). Does not mix scan orders across
    scale boundaries; scanning is within each group (B, L, D) using (Htok, Wtok).
    Merge is by sum (paper default). Spiral: top-right start, CCW, periphery→center.

    Input: (B, 3L, D). Output: (B, 3L, D).
    """

    def __init__(
        self,
        d_model: int,
        token_hw: Tuple[int, int],
        core: SequenceCore,
        *,
        conv_kernel: int = 3,
        merge: str = "sum",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_hw = tuple(token_hw)
        self.H, self.W = self.token_hw
        assert self.H > 0 and self.W > 0, "MDSSM: token_hw (H, W) must be positive"
        self.L = self.H * self.W
        self.core = core
        if merge != "sum":
            raise ValueError(f"MDSSM: only merge='sum' is implemented, got {merge!r}")
        self.merge_mode = merge

        # Eq. (8a): p branch — Linear + Conv1d (shared across scale groups)
        self.p_linear = nn.Linear(d_model, d_model)
        self.p_conv1d = nn.Conv1d(d_model, d_model, conv_kernel, padding=conv_kernel // 2)

        # Eq. (8c): g branch
        self.g_linear = nn.Linear(d_model, d_model)

        # Eq. (8b): merge paths — LayerNorm after sum
        self.merge_ln = nn.LayerNorm(d_model)

        # Eq. (8d): gated output projection
        self.out_linear = nn.Linear(d_model, d_model)

        # Scan indices (H, V, spiral) — register as buffers so they follow device
        idx_h = horizontal_indices(self.H, self.W)
        idx_v = vertical_indices(self.H, self.W)
        idx_s = spiral_indices_topright_ccw(self.H, self.W)
        self.register_buffer("_idx_h", idx_h)
        self.register_buffer("_idx_v", idx_v)
        self.register_buffer("_idx_s", idx_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3L, D). Returns (B, 3L, D). Preserves dtype/device; output is finite for finite input."""
        B, N, D = x.shape
        assert x.ndim == 3, f"MDSSM forward: x.ndim={x.ndim} (expect 3)"
        assert N == 3 * self.L, f"MDSSM forward: x.shape[1]={N} != 3*L={3*self.L}"
        assert D == self.d_model, f"MDSSM forward: x.shape[2]={D} != d_model={self.d_model}"

        xs, xc, xf = split_concat_sequence(x, self.L)

        def one_scale(xg: torch.Tensor) -> torch.Tensor:
            p = compute_p_branch(xg, self.p_linear, self.p_conv1d)
            g = compute_g_branch(xg, self.g_linear)
            px = run_scan_path(p, self._idx_h, self.core)
            py = run_scan_path(p, self._idx_v, self.core)
            pθ = run_scan_path(p, self._idx_s, self.core)
            sigma = merge_paths(px, py, pθ, self.merge_ln, self.merge_mode)
            return gated_output(sigma, g, self.out_linear)

        ys = one_scale(xs)
        yc = one_scale(xc)
        yf = one_scale(xf)
        y = merge_concat_sequence(ys, yc, yf)
        assert y.shape == x.shape
        return y
