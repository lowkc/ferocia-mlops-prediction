"""Configuration dataclasses for model training pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    """Configuration for training data paths.

    Attributes:
        train_path: Path to training dataset CSV file.
        test_path: Path to test dataset CSV file.
        target_column: Name of the target column.
    """

    train_path: Path = Path("data/processed/train.csv")
    test_path: Path = Path("data/processed/test.csv")
    target_column: str = "y"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Ensure paths are Path objects
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)


@dataclass
class FeatureConfig:
    """Configuration for feature preprocessing.

    Attributes:
        categorical_features: List of categorical feature names for one-hot encoding.
        numerical_features: List of numerical feature names for scaling.
        binary_features: List of binary feature names (no transformation needed).
    """

    categorical_features: list[str] = field(default_factory=list)
    numerical_features: list[str] = field(default_factory=list)
    binary_features: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for model training.

    Attributes:
        type: Model type (e.g., 'XGBClassifier').
        parameters: Dictionary of model hyperparameters.
    """

    type: str = "XGBClassifier"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model configuration."""
        supported_models = ["XGBClassifier"]
        if self.type not in supported_models:
            raise ValueError(
                f"Model type '{self.type}' not supported. "
                f"Supported models: {supported_models}"
            )


def load_config(config_path: str | Path) -> tuple[str, DataConfig, FeatureConfig, ModelConfig]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (job_name, DataConfig, FeatureConfig, ModelConfig) instances.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """

    def get_and_validate_dict(config_dict: Dict[str, Any], key: str) -> Dict:
        """Retrieve a key's value from config_dict and validate it's a dict."""
        value = config_dict.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"'{key}' section in YAML must be a dictionary")
        return value

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

    # Extract job name
    job_name = config_dict.get("job_name", "model_training")

    # Load data configuration
    data_config_dict = get_and_validate_dict(config_dict, "data")
    data_config = DataConfig(
        train_path=Path(data_config_dict.get("train_path", "data/processed/train.csv")),
        test_path=Path(data_config_dict.get("test_path", "data/processed/test.csv")),
        target_column=data_config_dict.get("target_column", "y"),
    )

    # Load preprocessing/feature configuration
    preprocessing_dict = get_and_validate_dict(config_dict, "preprocessing")
    feature_config = FeatureConfig(
        categorical_features=preprocessing_dict.get("categorical_features", []),
        numerical_features=preprocessing_dict.get("numerical_features", []),
        binary_features=preprocessing_dict.get("binary_features", []),
    )

    # Load model configuration
    model_dict = get_and_validate_dict(config_dict, "model")
    model_config = ModelConfig(
        type=model_dict.get("type", "XGBClassifier"),
        parameters=model_dict.get("parameters", {}),
    )

    return job_name, data_config, feature_config, model_config
