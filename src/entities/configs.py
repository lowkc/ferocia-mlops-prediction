"""Base configuration classes for pipelines"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from src.utils.config_utils import ensure_path


@dataclass
class BaseDataConfig:
    """Base configuration for handling config data across different pipelines.

    Attributes:
        target_column: Name of the target column in the dataset.
    """

    target_column: str = "y"


@dataclass
class PreprocessingDataConfig(BaseDataConfig):
    """Configuration for data preprocessing pipeline.

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

        self.raw_data_path = ensure_path(self.raw_data_path)
        self.output_dir = ensure_path(self.output_dir)


@dataclass
class TrainingDataConfig(BaseDataConfig):
    """Configuration for model training data paths.

    Attributes:
        train_path: Path to training dataset CSV file.
        test_path: Path to test dataset CSV file.
        encode_target: Whether to encode the target variable as 0/1.
    """

    train_path: Path = Path("data/processed/train.csv")
    test_path: Path = Path("data/processed/test.csv")
    encode_target: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        self.train_path = ensure_path(self.train_path)
        self.test_path = ensure_path(self.test_path)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps.

    Attributes:
        handle_missing: Whether to handle missing values by imputing with the mean.
        drop_duplicates: Whether to drop duplicate rows from the dataset.
        engineer_features: Whether to perform feature engineering.
        save_metadata: Whether to save preprocessing metadata as JSON.
    """

    handle_missing: bool = True
    drop_duplicates: bool = True
    engineer_features: bool = True
    save_metadata: bool = True


@dataclass
class FeatureConfig:
    """Configuration for feature preprocessing in training pipeline.

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
                f"Model type '{self.type}' not supported. Supported models: {supported_models}"
            )


@dataclass
class PreprocessingMetadata:
    """Metadata about preprocessing operations for reproducibility.

    Attributes:
        original_columns: List of column names in raw dataset.
        columns_after_processing: List of column names after preprocessing.
        engineered_features: List of newly created feature names.
        train_samples: Number of samples in training set.
        test_samples: Number of samples in test set.
    """

    original_columns: list[str] = field(default_factory=list)
    columns_after_processing: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)
    train_samples: int = 0
    test_samples: int = 0
