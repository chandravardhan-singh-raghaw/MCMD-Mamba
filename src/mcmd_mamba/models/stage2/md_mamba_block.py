"""
Eq. (7) Pre-norm residual MD-Mamba block: norm → MD-SSM → residual add; optional FFN.
"""

from typing import Optional

import torch
import torch.nn as nn

from .md_ssm import MDSSM


def residual_add(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Helper: x + delta (keeps code readable)."""
    assert x.shape == delta.shape, f"residual_add: x.shape={x.shape} delta.shape={delta.shape}"
    return x + delta


def assert_block_io(x: torch.Tensor, y: torch.Tensor, d_model: int) -> None:
    """Shape checks: x and y same shape, last dim d_model, no NaNs."""
    assert x.dim() == 3 and y.dim() == 3
    assert x.shape == y.shape, f"x.shape={x.shape} y.shape={y.shape}"
    assert x.shape[2] == d_model
    assert not torch.isnan(x).any() and not torch.isnan(y).any(), "NaNs in x or y"


class FFN(nn.Module):
    """Feed-forward: Linear → GELU → Linear → Dropout."""

    def __init__(
        self,
        d_model: int,
        mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * mult
        self.linear1 = nn.Linear(d_model, d_inner)
        self.linear2 = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)."""
        return self.dropout(self.linear2(torch.nn.functional.gelu(self.linear1(x))))


class MDMambaBlock(nn.Module):
    """
    Eq. (7) Pre-norm residual MD-Mamba block:
    (7a) T' = MD-SSM(LayerNorm(T)) + T
    (7b) Tout = FFN(LayerNorm(T')) + T'
    """

    def __init__(
        self,
        d_model: int,
        md_ssm: MDSSM,
        *,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.md_ssm = md_ssm
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, mult=ffn_mult, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3L, D). Returns (B, 3L, D). Eq. (7a) then (7b)."""
        # (7a) T' = MD-SSM(LN(T)) + T
        z = self.norm1(x)
        t_prime = residual_add(x, self.md_ssm(z))
        # (7b) Tout = FFN(LN(T')) + T'
        return residual_add(t_prime, self.ffn(self.norm2(t_prime)))
