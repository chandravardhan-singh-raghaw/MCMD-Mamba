"""Top-level MCMD-Mamba: Stage1 -> Stage2 -> Stage3 -> logits."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .stage1 import Stage1
from .stage1.mce import rgb_to_gray
from .stage2 import MDMambaStack
from .stage3 import Stage3


class MCMDMamba(nn.Module):
    """Full model wiring: Stage1 (MCE + PSA + tokenization) → Stage2 (MD-Mamba) → Stage3 (fusion + head)."""

    def __init__(self, cfg: Dict[str, Any], num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        model_cfg = cfg.get("model", cfg)
        m1 = model_cfg.get("stage1", {})
        m2 = model_cfg.get("stage2", {})
        m3 = model_cfg.get("stage3", {})

        # embed_dim / d_model / hidden_dim: top-level overrides stage-level
        d_model = model_cfg.get("embed_dim") or model_cfg.get("d_model") or model_cfg.get("hidden_dim") or m1.get("d_model", 256)
        d_model = int(d_model)
        feat_ch = m1.get("feat_ch", d_model)
        token_hw = tuple(model_cfg.get("token_hw") or m1.get("token_hw", (14, 14)))
        L = token_hw[0] * token_hw[1]

        # Stage1: MCE + PSA + tokenization
        self.stage1 = Stage1(
            feat_ch=feat_ch,
            d_model=d_model,
            token_hw=token_hw,
            patch_size=m1.get("patch_size", 8),
            sharpen=m1.get("sharpen", 4.0),
            cfg=m1,
        )

        # Stage2: MD-Mamba blocks
        self.stage2 = MDMambaStack(
            d_model=d_model,
            token_hw=token_hw,
            num_blocks=m2.get("num_blocks", 3),
            core_kind=m2.get("core_kind", "mamba"),
            conv_kernel=m2.get("conv_kernel", 3),
            merge=m2.get("merge", "sum"),
            dropout=m2.get("dropout", 0.1),
            cfg=m2,
        )

        # Stage3: weighted fusion + head
        self.stage3 = Stage3(
            embed_dim=d_model,
            num_classes=num_classes,
            tokens_per_scale=L,
            dropout=m3.get("dropout", 0.1),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_clahe_1ch: Optional[torch.Tensor] = None,
        *,
        return_intermediates: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        x: (B, 3, H, W) RGB input.
        x_clahe_1ch: (B, 1, H, W) optional; if None, uses grayscale fallback.
        return_intermediates: if True, return (logits, {"T": T, "T_out": T_out}).
        Returns: logits (B, num_classes), or (logits, intermediates) if return_intermediates.
        """
        if x_clahe_1ch is None:
            x_clahe_1ch = rgb_to_gray(x)
        T = self.stage1(x, x_clahe_1ch=x_clahe_1ch, check_shapes=True, **kwargs)
        T_out = self.stage2(T)
        logits = self.stage3(T_out)
        if return_intermediates:
            return logits, {"T": T, "T_out": T_out}
        return logits
