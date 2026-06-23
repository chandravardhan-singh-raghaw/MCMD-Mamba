"""AdamW optimizer and schedulers (cosine, warmup) for training."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR


def get_optimizer(
    params: List[torch.Tensor] | nn.ParameterList,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    name: str = "adamw",
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Build optimizer (default AdamW)."""
    if name.lower() == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    num_training_steps: int = 10000,
    warmup_steps: int = 500,
    **kwargs: Any,
) -> Optional[Any]:
    """Build LR scheduler (cosine with optional warmup)."""
    if name == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())
        return LambdaLR(optimizer, lr_lambda)
    return None
