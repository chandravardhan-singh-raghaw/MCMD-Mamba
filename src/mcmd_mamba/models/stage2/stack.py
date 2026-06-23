"""Stack K MD-Mamba blocks (K=3 in paper)."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .mamba_core import build_core
from .md_ssm import MDSSM
from .md_mamba_block import MDMambaBlock


class MDMambaStack(nn.Module):
    """Stack K MDMambaBlock layers. K=3 in paper. Input (B, 3L, D)."""

    def __init__(
        self,
        d_model: int,
        token_hw: Tuple[int, int],
        num_blocks: int = 3,
        *,
        core_kind: str = "mamba",
        conv_kernel: int = 3,
        merge: str = "sum",
        dropout: float = 0.1,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        cfg = cfg or {}
        self.d_model = d_model
        self.token_hw = tuple(token_hw)
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        kind = cfg.get("core_kind", core_kind)
        for _ in range(num_blocks):
            core = build_core(kind, d_model, **{k: v for k, v in cfg.items() if k != "core_kind"})
            md_ssm = MDSSM(
                d_model,
                token_hw=token_hw,
                core=core,
                conv_kernel=conv_kernel,
                merge=merge,
            )
            self.blocks.append(MDMambaBlock(d_model, md_ssm, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3L, D). Returns (B, 3L, D)."""
        for block in self.blocks:
            x = block(x)
        return x
