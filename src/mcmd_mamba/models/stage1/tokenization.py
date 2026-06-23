"""Downsample, flatten, project, scale-embeds: feature map → token sequence."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleEmbedding(nn.Module):
    """Placeholder: learnable scale/positional embedding added to token sequence."""

    def __init__(self, d_model: int, num_tokens: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        # Optional fixed length; if None, use broadcast (1, 1, d_model) for any N
        if num_tokens is not None:
            self.embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        else:
            self.embed = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D) with scale/pos embed added."""
        return x + self.embed  # (1,N,D) or (1,1,D) broadcasts to (B,N,D)


def concat_with_scale_embeddings(
    tokens: torch.Tensor,
    scale_embed: ScaleEmbedding,
) -> torch.Tensor:
    """Placeholder: add scale embedding to token sequence. tokens (B, N, D) → (B, N, D)."""
    return scale_embed(tokens)


class Tokenizer(nn.Module):
    """Resize to token_hw (optional), flatten, linear project. Output (B, L, d_model) with L = token_hw[0]*token_hw[1]."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        in_channels: int,
        d_model: int,
        token_hw: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        _hw = token_hw or cfg.get("token_hw")
        self.token_hw = tuple(_hw) if _hw is not None else None  # (H, W) for fixed L
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → (B, L, d_model). If token_hw set, L = token_hw[0]*token_hw[1]."""
        B, C, H, W = x.shape
        if self.token_hw is not None:
            th, tw = self.token_hw
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
            H, W = th, tw
        # (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)  # (B, L, d_model)
