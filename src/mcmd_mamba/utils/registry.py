"""Registry for models, datasets, losses (optional)."""

from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T")
REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to register a callable under name."""

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        REGISTRY[name] = fn
        return fn

    return deco


def get(name: str, **kwargs: Any) -> Any:
    """Instantiate or return registered callable by name."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown: {name}")
    return REGISTRY[name](**kwargs)
