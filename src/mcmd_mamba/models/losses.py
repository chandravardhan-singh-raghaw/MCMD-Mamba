"""Task-specific losses: multiclass (CE), multilabel (BCE)."""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multiclass CE. logits (B, C), targets (B,) long."""
    return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)


def bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multilabel BCE. logits (B, C), targets (B, C) float."""
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weight)


def build_loss(
    task: str = "multiclass",  # "multiclass" | "multilabel"
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    **kwargs: Any,
) -> nn.Module:
    """
    Build loss module for task.
    multiclass -> nn.CrossEntropyLoss
    multilabel -> nn.BCEWithLogitsLoss
    """
    if task == "multiclass":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing, reduction=reduction)
    if task == "multilabel":
        return nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction=reduction)
    raise ValueError(f"Unknown task: {task!r}. Use 'multiclass' or 'multilabel'.")


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task: str = "multiclass",
    loss_fn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute loss from logits and targets.
    Uses loss_fn if provided; otherwise builds from task.
    multiclass: targets (B,) long; multilabel: targets (B, C) float.
    """
    if loss_fn is None:
        loss_fn = build_loss(task, **kwargs)
    return loss_fn(logits, targets)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal loss for multiclass (softmax) or multilabel (sigmoid)."""
    # Multiclass: pt = softmax(logits)[class]; focal = -alpha * (1-pt)^gamma * log(pt)
    # TODO: implement or use existing focal
    raise NotImplementedError("focal_loss")
