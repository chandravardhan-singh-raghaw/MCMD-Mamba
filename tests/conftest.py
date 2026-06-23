"""Pytest conftest: ensure mcmd_mamba is importable (add src to path when run from repo root)."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
src = root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
