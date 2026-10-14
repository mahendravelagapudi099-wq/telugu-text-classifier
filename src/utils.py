"""Shared helpers: config loading and project paths."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config; defaults to configs/config.yaml at the project root."""
    resolved = Path(path) if path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    with open(resolved, encoding="utf-8") as f:
        return yaml.safe_load(f)
