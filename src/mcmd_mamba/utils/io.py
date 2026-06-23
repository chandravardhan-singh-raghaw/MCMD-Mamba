"""I/O helpers: save/load checkpoints, config, results."""

import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def save_checkpoint(path: str | Path, model: Any, optimizer: Any = None, epoch: int = 0, **kwargs: Any) -> None:
    """Save model state_dict and optional optimizer/epoch to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), "epoch": epoch, **kwargs}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: Any = None) -> Dict[str, Any]:
    """Load checkpoint dict from path."""
    return torch.load(path, map_location=map_location, weights_only=False)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Write dict to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file to dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Write dict to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
