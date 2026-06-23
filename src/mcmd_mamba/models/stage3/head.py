"""
Stage 3 head: LayerNorm + Dropout, Linear + SiLU, Linear classifier.
F = Dropout(LayerNorm(F_fuse)), H = SiLU(Linear(F)), logits = Linear(H).
"""

import torch
import torch.nn as nn


class Stage3Head(nn.Module):
    """
    Classification head: LayerNorm + Dropout, Linear + SiLU, Linear -> logits.
    Input: F_fuse (B, 2D). Output: logits (B, num_classes).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, embed_dim) e.g. F_fuse (B, 2D).
        Returns logits: (B, num_classes).
        """
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.nn.functional.silu(self.linear1(x))
        return self.classifier(x)
