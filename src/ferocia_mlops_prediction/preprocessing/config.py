"""Configuration dataclasses for data preprocessing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data preprocessing pipeline.

    Attributes:
        raw_data_path: Path to raw dataset CSV file.
        output_dir: Directory to save processed datasets.
        test_size: Proportion of data to use for test set (0.0 to 1.0).
        random_seed: Random seed for reproducibility.
        stratify: Whether to use stratified splitting based on target variable.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        save_metadata: Whether to save preprocessing metadata as JSON.
    """

    raw_data_path: Path = Path("data/dataset.csv")
    output_dir: Path = Path("data/processed")
    test_size: float = 0.2
    random_seed: int = 42
    stratify: bool = True
    log_level: str = "INFO"
    save_metadata: bool = True

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
class PreprocessingMetadata:
    """Metadata about preprocessing operations for reproducibility.

    Attributes:
        original_columns: List of column names in raw dataset.
        processed_columns: List of column names after preprocessing.
        binary_columns: Columns that were binary encoded.
        categorical_columns: Columns that were one-hot encoded.
        engineered_features: List of newly created feature names.
        target_column: Name of the target variable.
        train_samples: Number of samples in training set.
        test_samples: Number of samples in test set.
        test_size: Proportion used for test set.
        random_seed: Random seed used for splitting.
    """

    original_columns: list[str] = field(default_factory=list)
    processed_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    engineered_features: list[str] = field(default_factory=list)
    target_column: str = "y"
    train_samples: int = 0
    test_samples: int = 0
    test_size: float = 0.2
    random_seed: int = 42
