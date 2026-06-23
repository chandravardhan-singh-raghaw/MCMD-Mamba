"""Multi-Channel Enhancement: Gray, CLAHE, Sobel branches (Stage 1)."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# CLAHE (for dataset or optional in-forward fallback)
# ----------------------------

try:
    import cv2  # type: ignore[import-untyped]
except ImportError:
    cv2 = None  # type: ignore[assignment]


def clahe_1ch_from_rgb(
    x_rgb: torch.Tensor,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> torch.Tensor:
    """
    Apply CLAHE to RGB tensor and return 1-channel tensor.

    x_rgb: (B,3,H,W) float tensor in [0,1]
    returns: (B,1,H,W) float tensor in [0,1]
    """
    if cv2 is None:
        raise ImportError("opencv-python is required for CLAHE. Install with: pip install opencv-python")
    assert x_rgb.ndim == 4 and x_rgb.shape[1] == 3

    device = x_rgb.device
    dtype = x_rgb.dtype

    x_np = (x_rgb.detach().cpu().numpy() * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    outs = []
    for img in x_np:
        # img: (3,H,W) -> (H,W,3)
        img = img.transpose(1, 2, 0)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_clahe = clahe.apply(l_ch)
        lab_clahe = cv2.merge((l_clahe, a_ch, b_ch))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        gray = cv2.cvtColor(rgb_clahe, cv2.COLOR_RGB2GRAY)
        outs.append(gray)

    out = np.stack(outs, axis=0)  # (B,H,W)
    out = torch.from_numpy(out).unsqueeze(1).float() / 255.0
    return out.to(device=device, dtype=dtype)


# ----------------------------
# Simple transforms (GPU)
# ----------------------------

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) -> (B,1,H,W)."""
    w = torch.tensor(
        [0.2989, 0.5870, 0.1140],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def sobel_edges(x_gray: torch.Tensor) -> torch.Tensor:
    """x_gray: (B,1,H,W) -> (B,1,H,W) normalized magnitude."""
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=x_gray.device,
        dtype=x_gray.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=x_gray.device,
        dtype=x_gray.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x_gray, kx, padding=1)
    gy = F.conv2d(x_gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    mag = mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return mag


# ----------------------------
# Minimal backbone (replace with ConvNeXt adapter later)
# ----------------------------

class SimpleBranchCNN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stem_kernel: int) -> None:
        super().__init__()
        pad = stem_kernel // 2
        self.stem = nn.Conv2d(in_ch, 64, kernel_size=stem_kernel, stride=2, padding=pad)
        self.body = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)


# ----------------------------
# Multi-Channel Enhancement (MCE)
# ----------------------------

class MultiChannelEnhancement(nn.Module):
    """
    Stage 1: Multi-Channel Enhancement (MCE).
    Returns feature maps (Fs, Fc, Ff) for structural (Gray), contextual (CLAHE), fine (Sobel).
    Paper: Gray/CLAHE/Sobel + ConvNeXt variants with kernel 7/5/3.
    """

    def __init__(
        self,
        feat_ch: int = 256,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or {}
        feat_ch = self.cfg.get("feat_ch", feat_ch)
        self.structural = SimpleBranchCNN(in_ch=1, out_ch=feat_ch, stem_kernel=7)
        self.contextual = SimpleBranchCNN(in_ch=1, out_ch=feat_ch, stem_kernel=5)
        self.fine = SimpleBranchCNN(in_ch=1, out_ch=feat_ch, stem_kernel=3)
        self.feat_ch = feat_ch

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_clahe_1ch: Optional[torch.Tensor] = None,
        *,
        clahe_is_precomputed: bool = True,
        clahe_fallback_from_rgb: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x_rgb: (B,3,H,W) normalized to [0,1]
        x_clahe_1ch: (B,1,H,W) precomputed CLAHE from dataset (recommended for training).
        If None and clahe_is_precomputed=False (or clahe_fallback_from_rgb=True), CLAHE is computed from x_rgb (CPU/cv2).
        """
        x_gray = rgb_to_gray(x_rgb)
        x_sobel = sobel_edges(x_gray)

        if x_clahe_1ch is None:
            if (not clahe_is_precomputed or clahe_fallback_from_rgb) and cv2 is not None:
                x_clahe_1ch = clahe_1ch_from_rgb(
                    x_rgb,
                    clip_limit=clahe_clip_limit,
                    tile_grid_size=clahe_tile_grid_size,
                )
            else:
                raise ValueError(
                    "x_clahe_1ch must be provided (e.g. from dataset) or set clahe_fallback_from_rgb=True "
                    "with opencv-python installed."
                )

        Fs = self.structural(x_gray)       # (B, feat_ch, Hs, Ws)
        Fc = self.contextual(x_clahe_1ch)  # (B, feat_ch, Hc, Wc)
        Ff = self.fine(x_sobel)            # (B, feat_ch, Hf, Wf)

        return Fs, Fc, Ff
