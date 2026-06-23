"""
Scan index generation + reorder/undo utils for multi-direction SSM.

Tokens T ∈ ℝ^{L×D} (or (B,L,D)) come from a 2D grid Htok×Wtok with L = Htok*Wtok.
Each scan defines an index order; apply_index reorders tokens along the sequence;
invert_index gives the inverse for roundtrip.
"""

from typing import Optional, Union

import torch


def horizontal_indices(
    H: int,
    W: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Ψx: Row-major order (left→right, top→bottom).
    Same as flattening a CNN feature map.
    Preserves local horizontal continuity (e.g. elongated vessels, streak-like lesions).
    Index order: [0,1,2, 3,4,5, 6,7,8] for 3×3.
    """
    return torch.arange(H * W, device=device, dtype=torch.long)


def vertical_indices(
    H: int,
    W: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Ψy: Column-major order (top→bottom, column by column).
    Emphasizes vertical dependencies (e.g. vessel bifurcations, top-to-bottom progression).
    Index order: [0,3,6, 1,4,7, 2,5,8] for 3×3.
    """
    grid = torch.arange(H * W, device=device, dtype=torch.long).view(H, W)
    return grid.t().contiguous().view(-1)


def spiral_indices_topright_ccw(
    H: int,
    W: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Ψθ: Periphery→center, start top-right, counter-clockwise inward.
    Right column (top→bottom) → bottom row (right→left) → left column (bottom→top) → top row (left→right).
    Radial inductive bias; matches retinal anatomy (optic disc ≈ center).
    Index order e.g. 3×3: [2,5,8, 7,6, 3,0, 1, 4] (boundary first, center last).
    """
    top, left = 0, 0
    bottom, right = H - 1, W - 1
    order = []
    while top <= bottom and left <= right:
        # right column: top -> bottom
        for r in range(top, bottom + 1):
            order.append((r, right))
        right -= 1
        if left > right:
            break

        # bottom row: right -> left
        for c in range(right, left - 1, -1):
            order.append((bottom, c))
        bottom -= 1
        if top > bottom:
            break

        # left col: bottom -> top
        for r in range(bottom, top - 1, -1):
            order.append((r, left))
        left += 1
        if left > right:
            break

        # top row: left -> right
        for c in range(left, right + 1):
            order.append((top, c))
        top += 1

    flat = [r * W + c for (r, c) in order]
    return torch.tensor(flat, device=device, dtype=torch.long)


def apply_index(
    T: torch.Tensor,
    idx: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Reorder T along sequence dimension by index permutation.
    T: (B, L, D), idx: (L,) → (B, L, D) with T[:, idx, :].
    """
    idx = idx.to(T.device)
    return T.index_select(dim, idx)


def invert_index(idx: torch.Tensor) -> torch.Tensor:
    """
    Inverse permutation: apply_index(apply_index(T, idx), invert_index(idx)) == T.
    """
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype)
    return inv


# Backward-compat aliases (h, w as param names)
def horizontal_scan_indices(h: int, w: int) -> torch.Tensor:
    return horizontal_indices(h, w)


def vertical_scan_indices(h: int, w: int) -> torch.Tensor:
    return vertical_indices(h, w)


def spiral_scan_indices(h: int, w: int) -> torch.Tensor:
    return spiral_indices_topright_ccw(h, w)


def apply_scan(x: torch.Tensor, indices: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return apply_index(x, indices, dim)


def invert_scan(indices: torch.Tensor) -> torch.Tensor:
    return invert_index(indices)
