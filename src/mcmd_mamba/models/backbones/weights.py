"""Pretrained weight loading for ConvNeXt / backbone."""

from pathlib import Path
from typing import Any, Dict, Optional


def load_pretrained_weights(
    model: Any,
    path: Optional[str | Path] = None,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load state_dict from path (or default URL); optionally strict. Return missing/unexpected keys."""
    # TODO: load checkpoint, filter/by_name if needed, model.load_state_dict(...)
    raise NotImplementedError("load_pretrained_weights")
