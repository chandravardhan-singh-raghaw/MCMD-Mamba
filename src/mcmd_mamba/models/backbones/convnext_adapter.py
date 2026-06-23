"""Per-branch ConvNeXt config (kernel sizes 7, 5, 3) for MCE branches."""

from typing import Any, Dict, List, Optional

import torch.nn as nn


def build_convnext_branch(
    in_channels: int = 3,
    out_channels: int = 64,
    kernel_size: int = 7,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """Build a single ConvNeXt-style branch with given kernel size."""
    # TODO: use timm or custom ConvNeXt block(s) with specified kernel_size
    raise NotImplementedError("build_convnext_branch")


def build_convnext_branches(
    kernel_sizes: List[int] = [7, 5, 3],
    in_channels: int = 3,
    out_channels: int = 64,
    pretrained: bool = False,
    **kwargs: Any,
) -> nn.ModuleList:
    """Build one branch per kernel size (e.g. for MCE)."""
    return nn.ModuleList(
        [
            build_convnext_branch(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                pretrained=pretrained,
                **kwargs,
            )
            for k in kernel_sizes
        ]
    )
