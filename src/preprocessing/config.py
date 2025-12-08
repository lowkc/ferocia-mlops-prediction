"""Configuration dataclasses for data preprocessing pipeline."""

from pathlib import Path
from src.utils.config_utils import get_and_validate_dict, load_yaml_config
from src.entities.configs import PreprocessingDataConfig, PreprocessingConfig


def load_preprocessing_config(
    config_path: str | Path,
) -> tuple[PreprocessingDataConfig, PreprocessingConfig]:
    """Load preprocessing configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (PreprocessingDataConfig, PreprocessingConfig) instances.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_dict = load_yaml_config(config_path)

    data_config_dict = get_and_validate_dict(config_dict, "data")
    preprocessing_config_dict = get_and_validate_dict(config_dict, "preprocessing")

    data_config = PreprocessingDataConfig(
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
        drop_duplicates=preprocessing_config_dict.get("drop_duplicates", True),
        save_metadata=preprocessing_config_dict.get("save_metadata", True),
    )

    return data_config, preprocessing_config
