"""Seeds and determinism for reproducible runs."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True) -> None:
    """Enable PyTorch/CUDA determinism (may slow training)."""
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
