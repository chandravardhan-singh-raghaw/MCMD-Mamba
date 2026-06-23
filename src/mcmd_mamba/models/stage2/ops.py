"""
Core tensor utilities for Stage 2: split/merge scale groups, scan → core → unscan.
Everything else (MD-SSM, blocks) depends on these; keep shape asserts aggressive.
"""

from typing import Callable, Tuple

import torch

from .scans import apply_index, invert_index


def split_scales(T: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split T ∈ ℝ^{B×3L×D} → (Ts, Tc, Tf), each (B, L, D).
    """
    B, N, D = T.shape
    assert N == 3 * L, f"split_scales: T.shape[1]={N} != 3*L={3*L}"
    Ts = T[:, :L, :]
    Tc = T[:, L : 2 * L, :]
    Tf = T[:, 2 * L :, :]
    assert Ts.shape == (B, L, D) and Tc.shape == (B, L, D) and Tf.shape == (B, L, D)
    return Ts, Tc, Tf


def merge_scales(
    Ys: torch.Tensor,
    Yc: torch.Tensor,
    Yf: torch.Tensor,
) -> torch.Tensor:
    """
    Merge (Ys, Yc, Yf) → Y ∈ ℝ^{B×3L×D}.
    """
    assert Ys.dim() == 3 and Yc.dim() == 3 and Yf.dim() == 3
    assert Ys.shape == Yc.shape == Yf.shape, (
        f"merge_scales: shapes must match, got Ys={Ys.shape} Yc={Yc.shape} Yf={Yf.shape}"
    )
    return torch.cat([Ys, Yc, Yf], dim=1)


def scan_process_unscan(
    x: torch.Tensor,
    idx: torch.Tensor,
    core: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    apply scan → core → unscan.
    x: (B, L, D), idx: (L,) permutation, core: (B, L, D) → (B, L, D).
    Returns (B, L, D) in original (unscanned) order.
    """
    B, L, D = x.shape
    assert idx.shape == (L,), f"scan_process_unscan: idx.shape={idx.shape} != (L,)={L}"
    inv = invert_index(idx)
    x_scanned = apply_index(x, idx, dim=1)  # (B, L, D) in scan order
    y_scanned = core(x_scanned)               # (B, L, D) in scan order
    assert y_scanned.shape == (B, L, D), f"core must preserve shape, got {y_scanned.shape}"
    y = apply_index(y_scanned, inv, dim=1)    # (B, L, D) back to original order
    return y


# Optional: Conv1D, SiLU, gating for SSM blocks (used by mamba_core / real Mamba later)
def conv1d_silu(x: torch.Tensor, conv: "torch.nn.Conv1d") -> torch.Tensor:
    """Apply Conv1d then SiLU."""
    return torch.nn.functional.silu(conv(x))


def gate_linear(
    x: torch.Tensor,
    gate: "torch.nn.Linear",
    proj: "torch.nn.Linear",
) -> torch.Tensor:
    """Element-wise gate * proj (e.g. for SSM input modulation)."""
    return gate(x) * proj(x)
