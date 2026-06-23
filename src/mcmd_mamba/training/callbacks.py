"""Checkpointing, early stopping, and logging callbacks for training."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional


class CheckpointCallback:
    """Save best K checkpoints by monitored metric."""

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val/accuracy",
        mode: str = "max",
        save_top_k: int = 2,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_k: list = []

    def __call__(self, epoch: int, metrics: Dict[str, float], model: Any, optimizer: Any) -> None:
        # TODO: compare metrics[self.monitor], keep best_k, save state_dict
        pass


class EarlyStoppingCallback:
    """Stop when monitored metric does not improve for patience epochs."""

    def __init__(self, monitor: str = "val/accuracy", mode: str = "max", patience: int = 15) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best: Optional[float] = None
        self.wait = 0

    def __call__(self, metrics: Dict[str, float]) -> bool:
        """Return True if should stop."""
        val = metrics.get(self.monitor)
        if val is None:
            return False
        if self.best is None:
            self.best = val
            return False
        if self.mode == "max" and val > self.best:
            self.best = val
            self.wait = 0
            return False
        if self.mode == "min" and val < self.best:
            self.best = val
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience
