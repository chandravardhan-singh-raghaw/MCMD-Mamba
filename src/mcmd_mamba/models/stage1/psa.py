"""Pyramidal Self-Attention mask over spatial dimensions."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidalSelfAttention(nn.Module):
    """
    PSA: builds a spatial mask from fine-grained feature map and reweights feature maps.
    Output mask M: (B,1,Hf,Wf), then downsample to match other scales.
    """

    def __init__(
        self,
        patch_size: int = 8,
        sharpen: float = 4.0,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self.patch_size = self.cfg.get("patch_size", patch_size)
        self.sharpen = self.cfg.get("sharpen", sharpen)

    def forward(self, Ff: torch.Tensor) -> torch.Tensor:
        """
        Ff: (B,C,H,W) fine-grained feature map (e.g. from Sobel branch).
        Returns: M (B,1,Hf,Wf) spatial mask.
        """
        A = Ff.max(dim=1, keepdim=True).values  # (B,1,H,W)

        ps = self.patch_size
        A_p = F.max_pool2d(A, kernel_size=ps, stride=ps, ceil_mode=True)  # patch-max

        # normalize per image
        amin = A_p.amin(dim=(-2, -1), keepdim=True)
        amax = A_p.amax(dim=(-2, -1), keepdim=True)
        A_p = (A_p - amin) / (amax - amin + 1e-6)

        M = F.interpolate(A_p, size=Ff.shape[-2:], mode="bilinear", align_corners=False)
        M = torch.sigmoid(self.sharpen * (M - 0.5))
        return M  # (B,1,Hf,Wf)


def apply_psa(M: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """
    Downsample mask to match feature map and apply multiplicatively.
    M: (B,1,Hm,Wm), feat: (B,C,H,W) -> (B,C,H,W).
    """
    M_resized = F.interpolate(M, size=feat.shape[-2:], mode="bilinear", align_corners=False)
    return feat * M_resized
