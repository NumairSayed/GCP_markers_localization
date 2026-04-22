"""Config loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """dict subclass that also supports attribute access for convenience."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)

def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    elif hasattr(obj, "items"):   # custom mapping like Config
        return {k: to_builtin(v) for k, v in obj.items()}
    else:
        return obj

def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.safe_dump(
            to_builtin(cfg),
            f,
            sort_keys=False
        )
