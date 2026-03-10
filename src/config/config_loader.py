"""Load and validate configuration from config.yaml."""

from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default relative to this file.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
