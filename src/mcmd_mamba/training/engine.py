"""Training loop: forward, loss, backward, AMP, grad clip."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _unpack_batch(batch: Any) -> tuple:
    """Unpack batch into (x, y). Supports (x,y), dict with image/label, etc."""
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    x = batch.get("image", batch.get("x"))
    y = batch.get("label", batch.get("y", batch.get("target")))
    return x, y


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[Any] = None,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Run one epoch; return dict of loss. Uses logits from model, loss_fn(logits, targets)."""
    model.train()
    loss_fn = loss_fn.to(device)
    total_loss = 0.0
    n = 0

    for batch in loader:
        x, y = _unpack_batch(batch)
        if x is None or y is None:
            continue
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return {"loss": total_loss / max(n, 1)}


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Run validation; return loss and optionally metrics."""
    model.eval()
    loss_fn = loss_fn.to(device)
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in loader:
            x, y = _unpack_batch(batch)
            if x is None or y is None:
                continue
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)

    return {"loss": total_loss / max(n, 1)}
