"""Load config from YAML (OmegaConf or plain yaml). Optional Hydra composition."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a single YAML file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. base is mutated."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load config from path; optionally merge dataset and model configs.
    Optionally apply overrides (e.g. from CLI).
    """
    cfg = load_yaml(path)
    if dataset_path:
        ds_cfg = load_yaml(dataset_path)
        cfg.setdefault("dataset", {}).update(ds_cfg)
    if model_path:
        m_cfg = load_yaml(model_path)
        cfg.setdefault("model", {}).update(m_cfg.get("model", m_cfg))
    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
    return cfg


def get_num_classes(cfg: Dict[str, Any]) -> int:
    """Get num_classes from cfg.dataset or cfg.num_classes."""
    ds = cfg.get("dataset", cfg)
    n = ds.get("num_classes")
    if n is not None:
        return int(n)
    return int(cfg.get("num_classes", 5))


def get_embed_dim(cfg: Dict[str, Any]) -> int:
    """Get embed_dim / d_model / hidden_dim from model config."""
    m = cfg.get("model", cfg)
    return int(m.get("embed_dim") or m.get("d_model") or m.get("hidden_dim", 256))


def get_token_hw(cfg: Dict[str, Any]) -> tuple:
    """Get token_hw from model config; L = H*W (deterministic)."""
    m = cfg.get("model", cfg)
    thw = m.get("token_hw") or m.get("stage1", {}).get("token_hw", [14, 14])
    return tuple(thw[:2]) if isinstance(thw, (list, tuple)) else (14, 14)


def get_tokens_per_scale(cfg: Dict[str, Any]) -> int:
    """Infer L = token_hw[0] * token_hw[1]."""
    h, w = get_token_hw(cfg)
    return h * w


def get_task(cfg: Dict[str, Any]) -> str:
    """Get task: multiclass or multilabel."""
    ds = cfg.get("dataset", cfg)
    t = ds.get("task", "multiclass")
    return "multiclass" if t == "multiclass" else "multilabel"


def build_model_from_config(cfg: Dict[str, Any]):
    """Create MCMDMamba from merged config. num_classes from dataset."""
    from ..models.mcmd_mamba import MCMDMamba

    num_classes = get_num_classes(cfg)
    return MCMDMamba(cfg, num_classes=num_classes)
