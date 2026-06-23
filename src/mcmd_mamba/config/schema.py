"""Config validation: required keys and types for train, dataset, and model sections."""

from typing import Any, Dict, List


def validate_train(cfg: Dict[str, Any]) -> List[str]:
    """Return list of validation errors for train section."""
    errors = []
    train = cfg.get("train", {})
    if not isinstance(train.get("batch_size"), int):
        errors.append("train.batch_size must be int")
    if not isinstance(train.get("max_epochs"), int):
        errors.append("train.max_epochs must be int")
    return errors


def validate_dataset(cfg: Dict[str, Any]) -> List[str]:
    """Return list of validation errors for dataset section."""
    errors = []
    ds = cfg.get("dataset", cfg)
    num_classes = ds.get("num_classes")
    if num_classes is not None and not isinstance(num_classes, int):
        errors.append("dataset.num_classes must be int")
    task = ds.get("task")
    if task is not None and task not in ("multiclass", "multilabel"):
        errors.append("dataset.task must be 'multiclass' or 'multilabel'")
    return errors


def validate_model(cfg: Dict[str, Any]) -> List[str]:
    """Return list of validation errors for model section."""
    errors = []
    m = cfg.get("model", cfg)
    embed_dim = m.get("embed_dim") or m.get("d_model") or m.get("hidden_dim")
    if embed_dim is not None and not isinstance(embed_dim, (int, float)):
        errors.append("model.embed_dim / d_model / hidden_dim must be int")
    token_hw = m.get("token_hw") or m.get("stage1", {}).get("token_hw")
    if token_hw is not None:
        if not isinstance(token_hw, (list, tuple)) or len(token_hw) < 2:
            errors.append("model.token_hw must be [H, W]")
    opt = cfg.get("optimizer", cfg)
    if opt.get("lr") is not None and not isinstance(opt.get("lr"), (int, float)):
        errors.append("optimizer.lr must be number")
    sched = cfg.get("scheduler", cfg)
    if sched.get("warmup_epochs") is not None and not isinstance(sched.get("warmup_epochs"), int):
        errors.append("scheduler.warmup_epochs must be int")
    return errors


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate full config; return all errors."""
    errors = []
    errors.extend(validate_train(cfg))
    errors.extend(validate_dataset(cfg))
    errors.extend(validate_model(cfg))
    return errors
