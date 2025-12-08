"""Configuration dataclasses for model training pipeline."""

from pathlib import Path

from src.utils.config_utils import get_and_validate_dict, load_yaml_config
from src.entities.configs import TrainingDataConfig, FeatureConfig, ModelConfig, TuningConfig


def load_training_config(
    config_path: str | Path,
) -> tuple[str, TrainingDataConfig, FeatureConfig, ModelConfig]:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of (job_name, DataConfig, FeatureConfig, ModelConfig) instances.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If the YAML structure is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_dict = load_yaml_config(config_path)

    # Extract job name
    job_name = config_dict.get("job_name", "model_training")

    # Load data configuration
    data_config_dict = get_and_validate_dict(config_dict, "data")
    data_config = TrainingDataConfig(
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


def load_tuning_config(
    tuning_config_path: str | Path,
    training_config_path: str | Path,
) -> tuple[str, TrainingDataConfig, FeatureConfig, ModelConfig, TuningConfig]:
    """Load tuning configuration from YAML files.

    This function loads both the training configuration (for model type, features, and
    fixed parameters) and tuning configuration (for hyperparameter search spaces).

    Args:
        tuning_config_path: Path to the tuning YAML configuration file.
        training_config_path: Path to the training YAML configuration file.

    Returns:
        Tuple of (job_name, TrainingDataConfig, FeatureConfig, ModelConfig, TuningConfig).

    Raises:
        FileNotFoundError: If either config file doesn't exist.
        ValueError: If the YAML structure is invalid.
        yaml.YAMLError: If the YAML is malformed.
    """
    # Load training config for base model setup
    job_name, data_config, feature_config, model_config = load_training_config(training_config_path)

    # Load tuning config for hyperparameter search spaces
    tuning_dict = load_yaml_config(tuning_config_path)

    # Extract study configuration
    study_dict = get_and_validate_dict(tuning_dict, "study")
    tuning_config = TuningConfig(
        direction=study_dict.get("direction", "maximize"),
        n_trials=study_dict.get("n_trials", 50),
        random_state=study_dict.get("random_state", 42),
        params=tuning_dict.get("params", {}),
    )

    return job_name, data_config, feature_config, model_config, tuning_config
