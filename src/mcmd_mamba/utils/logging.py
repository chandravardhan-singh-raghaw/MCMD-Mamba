"""Logging setup and helpers."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> None:
    """Configure root logger to stdout."""
    fmt = format_string or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)


def get_logger(name: str) -> logging.Logger:
    """Return logger with given name."""
    return logging.getLogger(name)
