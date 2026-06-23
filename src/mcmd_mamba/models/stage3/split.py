"""
Stage 3 split: partition T_out (B, 3L, D) into Ts, Tc, Tf (each B, L, D).
Tokens are ordered as [T_s || T_c || T_f] (structural, contextual, fine-grained).
"""

from typing import Tuple

import torch


def split_scales(
    T: torch.Tensor,
    L: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split T_out (B, 3L, D) into Ts, Tc, Tf (each B, L, D).
    Order: [Ts || Tc || Tf].
    """
    B, N, D = T.shape
    assert N == 3 * L, f"split_scales: T.shape[1]={N} != 3*L={3*L}"
    Ts = T[:, :L, :]
    Tc = T[:, L : 2 * L, :]
    Tf = T[:, 2 * L :, :]
    assert Ts.shape == (B, L, D) and Tc.shape == (B, L, D) and Tf.shape == (B, L, D)
    return Ts, Tc, Tf


def infer_L(T: torch.Tensor) -> int:
    """
    Infer L from T (B, 3L, D). Assumes 3 scale groups of equal length.
    L = T.shape[1] // 3.
    """
    N = T.shape[1]
    assert N % 3 == 0, f"infer_L: T.shape[1]={N} must be divisible by 3"
    return N // 3
