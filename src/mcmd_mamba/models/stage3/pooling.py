"""
Stage 3 pooling: dual pool (avg + max) over token dimension.
Keeps feature dim D; output (B, 2D) per scale group.
"""

import torch


def dual_pool_tokens(Ti: torch.Tensor) -> torch.Tensor:
    """
    Dual pooling over token dim: avg_pool + max_pool, concatenate.
    Ti: (B, L, D) -> P_i: (B, 2D).
    AP_i = mean(Ti, dim=1) -> (B, D)
    MP_i = max(Ti, dim=1)  -> (B, D)
    P_i = concat([AP_i, MP_i]) -> (B, 2D)
    """
    assert Ti.ndim == 3, f"dual_pool_tokens: Ti.ndim={Ti.ndim} (expect 3)"
    B, L, D = Ti.shape
    avg_pool = Ti.mean(dim=1)  # (B, D)
    max_pool = Ti.max(dim=1).values  # (B, D)
    return torch.cat([avg_pool, max_pool], dim=1)  # (B, 2D)
