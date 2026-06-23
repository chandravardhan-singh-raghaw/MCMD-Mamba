"""
Stage 3 weighted fusion: learnable ω_s, ω_c, ω_f (softmax normalized).
F_fuse = Σ_i ω_i * P_i where P_i = dual_pool(T_i).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .pooling import dual_pool_tokens


class WeightedFusion(nn.Module):
    """
    Learnable weights ω_s, ω_c, ω_f (normalized via softmax).
    forward(Ts, Tc, Tf) -> F_fuse (B, 2D).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        # P_i = dual_pool(T_i) -> (B, 2D) per scale; 3 scales
        self.weights = nn.Parameter(torch.ones(3))  # ω_s, ω_c, ω_f

    def forward(
        self,
        Ts: torch.Tensor,
        Tc: torch.Tensor,
        Tf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ts, Tc, Tf: each (B, L, D).
        Returns F_fuse: (B, 2D).
        """
        P_s = dual_pool_tokens(Ts)  # (B, 2D)
        P_c = dual_pool_tokens(Tc)
        P_f = dual_pool_tokens(Tf)

        w = torch.softmax(self.weights, dim=0)  # (3,)
        F_fuse = w[0] * P_s + w[1] * P_c + w[2] * P_f  # (B, 2D)
        return F_fuse
