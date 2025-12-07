"""Utilities for handling YAML configuration files."""

from typing import Any, Dict
from pathlib import Path
import yaml


def get_and_validate_dict(config_dict: Dict[str, Any], key: str) -> Dict:
    """Retrieve a key's value from config_dict and validate it's a dict."""
    value = config_dict.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' section in YAML must be a dictionary")
    return value


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid (not a dictionary).
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, mode="r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}: expected a dictionary")

    return config_dict


def ensure_path(path: str | Path) -> Path:
    """Convert string or Path to Path object.

    Args:
        path: String or Path object to convert.

    Returns:
        Path object.
    """
    return Path(path)
