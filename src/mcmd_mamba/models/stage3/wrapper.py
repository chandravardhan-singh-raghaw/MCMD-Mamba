"""Stage 3: split T_out -> dual pool -> weighted fusion -> head -> logits."""

import torch
import torch.nn as nn

from .split import infer_L, split_scales
from .weighted_fusion import WeightedFusion
from .head import Stage3Head


class Stage3(nn.Module):
    """
    Stage 3: split T_out (B, 3L, D) -> pool (dual) per scale -> weighted fusion -> head.
    Output: logits (B, num_classes).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        tokens_per_scale: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.tokens_per_scale = tokens_per_scale
        # Dual pool: (B, L, D) -> (B, 2D) per scale
        # F_fuse = Σ ω_i * P_i -> (B, 2D)
        self.fusion = WeightedFusion(d_model=embed_dim)
        # Head: F_fuse (B, 2D) -> logits (B, num_classes)
        self.head = Stage3Head(
            embed_dim=2 * embed_dim,  # F_fuse has 2D
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, T_out: torch.Tensor) -> torch.Tensor:
        """
        T_out: (B, 3L, D).
        Returns logits: (B, num_classes).
        """
        L = self.tokens_per_scale
        Ts, Tc, Tf = split_scales(T_out, L)
        F_fuse = self.fusion(Ts, Tc, Tf)  # (B, 2D)
        return self.head(F_fuse)
