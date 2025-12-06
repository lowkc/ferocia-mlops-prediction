"""Configuration dataclasses for data preprocessing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class DataConfig:
    """Configuration for data preprocessing pipeline. Reads from preprocessing YAML file.

    Attributes:
        raw_data_path: Path to raw dataset CSV file.
        output_dir: Directory to save processed datasets.
        test_size: Proportion of data to use for test set (0.0 to 1.0).
        random_seed: Random seed for reproducibility.
        stratify: Whether to use stratified splitting based on target variable.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    raw_data_path: Path = Path("data/raw/dataset.csv")
    output_dir: Path = Path("data/processed")
    target_column: str = "y"
    test_size: float = 0.2
    random_seed: int = 42
    stratify: bool = True
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")

        if self.random_seed < 0:
            raise ValueError(f"random_seed must be non-negative, got {self.random_seed}")

        # Ensure paths are Path objects
        self.raw_data_path = Path(self.raw_data_path)
        self.output_dir = Path(self.output_dir)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps.

    Attributes:
        handle_missing: Whether to handle missing values by imputing with the mean.
        drop_duplicates: Whether to drop duplicate rows from the dataset.
        engineer_features: Whether to perform feature engineering.
        encode_categoricals: Whether to encode categorical variables.
        log_transform_threshold: Threshold for log-transforming numerical features.
        save_metadata: Whether to save preprocessing metadata as JSON.
    """

    handle_missing: bool = True
    drop_duplicates: bool = True
    engineer_features: bool = True
    encode_categoricals: bool = True
    log_transform_threshold: float = 1.0
    save_metadata: bool = True


@dataclass
class PreprocessingMetadata:
    """Metadata about preprocessing operations for reproducibility.

    Attributes:
        original_columns: List of column names in raw dataset.
        columns_after_processing: List of column names after preprocessing.
        engineered_features: List of newly created feature names.
    """

    original_columns: list[str] = field(default_factory=list)
    columns_after_processing: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)


def load_config(config_path: str | Path) -> tuple[DataConfig, PreprocessingConfig]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (DataConfig, PreprocessingConfig) instances.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """

    def get_and_validate_dict(config_dict: Dict, key: str) -> None:
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

    data_config_dict = get_and_validate_dict(config_dict, "data")
    preprocessing_config_dict = get_and_validate_dict(config_dict, "preprocessing")

    data_config = DataConfig(
        raw_data_path=Path(data_config_dict.get("raw_path", "data/dataset.csv")),
        output_dir=Path(data_config_dict.get("processed_dir", "data/processed")),
        target_column=data_config_dict.get("target_column", "y"),
        test_size=data_config_dict.get("test_size", 0.2),
        random_seed=data_config_dict.get("random_state", 42),
        stratify=data_config_dict.get("stratify", True),
    )
    preprocessing_config = PreprocessingConfig(
        handle_missing=preprocessing_config_dict.get("handle_missing", True),
        engineer_features=preprocessing_config_dict.get("engineer_features", True),
        encode_categoricals=preprocessing_config_dict.get("encode_categoricals", True),
        log_transform_threshold=preprocessing_config_dict.get("log_transform_threshold", 1.0),
    )

    return data_config, preprocessing_config
