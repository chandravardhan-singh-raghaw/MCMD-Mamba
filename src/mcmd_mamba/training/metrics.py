"""Accuracy, F1 (macro/micro) for multiclass and multilabel."""

from typing import Optional

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> torch.Tensor:
    """Multiclass accuracy (top-k). logits (B, C), targets (B,) long."""
    _, pred = logits.topk(top_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return correct[:top_k].reshape(-1).float().sum() / targets.size(0)


def macro_f1_multiclass(
    pred: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> torch.Tensor:
    """Macro F1 for multiclass. pred/targets (B,) long."""
    from sklearn.metrics import f1_score
    import numpy as np
    p = pred.cpu().numpy()
    t = targets.cpu().numpy()
    return torch.tensor(f1_score(t, p, average=average, zero_division=0), device=pred.device)


def macro_f1_multilabel(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Macro F1 over labels. logits/targets (B, L)."""
    from sklearn.metrics import f1_score
    import numpy as np
    pred = (logits.sigmoid() > threshold).float().cpu().numpy()
    t = targets.cpu().numpy()
    return torch.tensor(f1_score(t, pred, average="macro", zero_division=0), device=logits.device)
