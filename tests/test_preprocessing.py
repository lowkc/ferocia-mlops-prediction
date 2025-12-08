"""Unit tests for data preprocessing pipeline."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from entities.configs import PreprocessingDataConfig, PreprocessingConfig

from preprocessing.config import load_preprocessing_config
from preprocessing.data_preprocessing import (
    DataCleaner,
    DataLoader,
    DataSplitter,
    FeatureEngineer,
    PreprocessingPipeline,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading configuration from valid YAML file."""
        # Create a valid YAML config file
        config_path = tmp_path / "test_config.yaml"
        config_data = {
            "data": {
                "raw_path": "data/test/input.csv",
                "processed_dir": "data/test/output",
                "target_column": "target",
                "test_size": 0.25,
                "random_state": 123,
                "stratify": False,
            },
            "preprocessing": {
                "handle_missing": False,
                "drop_duplicates": True,
                "engineer_features": True,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        data_config, preprocessing_config = load_preprocessing_config(config_path)

        # Verify PreprocessingDataConfig
        assert data_config.raw_data_path == Path("data/test/input.csv")
        assert data_config.output_dir == Path("data/test/output")
        assert data_config.target_column == "target"
        assert data_config.test_size == 0.25
        assert data_config.random_seed == 123
        assert data_config.stratify is False

        # Verify PreprocessingConfig
        assert preprocessing_config.handle_missing is False
        assert preprocessing_config.engineer_features is True
        assert preprocessing_config.drop_duplicates is True

    def test_load_config_with_defaults(self, tmp_path):
        """Test loading configuration with missing optional fields uses defaults."""
        config_path = tmp_path / "minimal_config.yaml"
        config_data = {
            "data": {
                "raw_path": "data/input.csv",
            },
            "preprocessing": {},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        data_config, preprocessing_config = load_preprocessing_config(config_path)

        # Verify defaults are used
        assert data_config.raw_data_path == Path("data/input.csv")
        assert data_config.output_dir == Path("data/processed")  # default
        assert data_config.test_size == 0.2  # default
        assert preprocessing_config.handle_missing is True  # default
        assert preprocessing_config.engineer_features is True  # default

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_preprocessing_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test error with malformed YAML."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content:\n  - broken")

        with pytest.raises(yaml.YAMLError, match="Error parsing YAML file"):
            load_preprocessing_config(config_path)

    def test_load_config_not_dict(self, tmp_path):
        """Test error when YAML is not a dictionary."""
        config_path = tmp_path / "list_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(["item1", "item2"], f)

        with pytest.raises(ValueError, match="Invalid YAML structure.*expected a dictionary"):
            load_preprocessing_config(config_path)

    def test_load_config_data_not_dict(self, tmp_path):
        """Test error when 'data' section is not a dictionary."""
        config_path = tmp_path / "bad_data.yaml"
        config_data = {"data": "not a dict", "preprocessing": {}}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="'data' section in YAML must be a dictionary"):
            load_preprocessing_config(config_path)

    def test_load_config_triggers_dataconfig_validation(self, tmp_path):
        """Test that invalid values trigger PreprocessingDataConfig validation errors."""
        config_path = tmp_path / "invalid_values.yaml"
        config_data = {
            "data": {
                "raw_path": "data.csv",
                "test_size": 1.5,  # Invalid: must be between 0 and 1
            },
            "preprocessing": {},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Should raise ValueError from PreprocessingDataConfig.__post_init__
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            load_preprocessing_config(config_path)

    def test_load_config_invalid_random_seed(self, tmp_path):
        """Test that negative random_seed triggers validation error."""
        config_path = tmp_path / "invalid_seed.yaml"
        config_data = {
            "data": {
                "raw_path": "data.csv",
                "random_state": -10,  # Invalid: must be non-negative
            },
            "preprocessing": {},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="random_seed must be non-negative"):
            load_preprocessing_config(config_path)


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        # Create sample CSV file
        csv_path = tmp_path / "test.csv"
        df_sample = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df_sample.to_csv(csv_path, sep=";", index=False)

        # Load data
        logger = Mock()
        loader = DataLoader(logger)
        df = loader.load_data(csv_path)

        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2"]
        logger.info.assert_called()

    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        logger = Mock()
        loader = DataLoader(logger)

        with pytest.raises(FileNotFoundError):
            loader.load_data(Path("nonexistent.csv"))


class TestDataCleaner:
    """Tests for DataCleaner class."""

    def test_check_missing_values_none(self):
        """Test missing value check with no missing data."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        logger = Mock()
        cleaner = DataCleaner(logger)
        df_result = cleaner.check_missing_values(df)

        assert df.equals(df_result)
        assert any("No missing values" in str(call) for call in logger.info.call_args_list)

    def test_check_missing_values_present(self):
        """Test missing value check with missing data."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": ["a", "b", None]})

        logger = Mock()
        cleaner = DataCleaner(logger)
        df_result = cleaner.check_missing_values(df)

        assert df.equals(df_result)
        logger.warning.assert_called()

    def test_impute_missing_values_with_mean(self):
        """Test imputation of missing values with mean."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [4, 5, np.nan]})

        logger = Mock()
        cleaner = DataCleaner(logger)
        df_result = cleaner.impute_missing_values_with_mean(df)

        assert df_result["col1"].tolist() == [1.0, 2.0, 3.0]
        assert df_result["col2"].tolist() == [4.0, 5.0, 4.5]
        logger.info.assert_called()


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_transform_pdays(self):
        """Test pdays transformation."""
        df = pd.DataFrame({"pdays": [-1, 5, -1, 10, 15]})

        logger = Mock()
        engineer = FeatureEngineer(logger)
        df_transformed = engineer.transform_pdays(df)

        assert "pdays" not in df_transformed.columns
        assert "previous_contact" in df_transformed.columns
        assert "days_since_last_contact" in df_transformed.columns

        # Check values
        assert df_transformed["previous_contact"].tolist() == [0, 1, 0, 1, 1]
        assert df_transformed["days_since_last_contact"].tolist() == [0, 5, 0, 10, 15]

    def test_total_contacts(self):
        """Test total_contacts feature creation."""
        df = pd.DataFrame({"campaign": [1, 2, 3], "previous": [0, 1, 2]})

        logger = Mock()
        engineer = FeatureEngineer(logger)
        df_transformed = engineer.total_contacts(df)
        assert "total_contacts" in df_transformed.columns
        assert df_transformed["total_contacts"].tolist() == [1, 3, 5]


class TestDataSplitter:
    """Tests for DataSplitter class."""

    def test_split_data(self):
        """Test train/test splitting."""
        df = pd.DataFrame(
            {"feature1": range(100), "feature2": range(100, 200), "target": [0, 1] * 50}
        )

        logger = Mock()
        splitter = DataSplitter(logger)
        x_train, x_test, y_train, y_test = splitter.split_data(
            df, target_column="target", test_size=0.2, random_seed=42, stratify=True
        )

        assert len(x_train) == 80
        assert len(x_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert "target" not in x_train.columns
        assert "target" not in x_test.columns

    def test_split_data_missing_target(self):
        """Test error handling for missing target column."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})

        logger = Mock()
        splitter = DataSplitter(logger)

        with pytest.raises(ValueError, match="Target column 'target' not found"):
            splitter.split_data(df, target_column="target", test_size=0.2, random_seed=42)


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline class."""

    @patch("preprocessing.data_preprocessing.DataLoader.load_data")
    def test_pipeline_run(self, mock_load_data, tmp_path):
        """Test full pipeline execution."""
        # Create sample data
        sample_data = pd.DataFrame(
            {
                "age": [30, 40, 50, 60] * 5,
                "job": ["admin", "technician", "services", "admin"] * 5,
                "marital": ["married", "single", "divorced", "married"] * 5,
                "education": ["tertiary", "secondary", "primary", "tertiary"] * 5,
                "default": ["no", "no", "yes", "no"] * 5,
                "balance": [1000, 2000, 3000, 4000] * 5,
                "housing": ["yes", "no", "yes", "no"] * 5,
                "loan": ["no", "yes", "no", "no"] * 5,
                "contact": ["cellular", "telephone", "cellular", "cellular"] * 5,
                "day": [15, 16, 17, 18] * 5,
                "month": ["may", "jun", "jul", "aug"] * 5,
                "duration": [100, 200, 300, 400] * 5,
                "campaign": [1, 2, 1, 3] * 5,
                "pdays": [-1, 5, -1, 10] * 5,
                "previous": [0, 1, 0, 2] * 5,
                "poutcome": ["unknown", "success", "failure", "unknown"] * 5,
                "y": ["no", "yes", "no", "yes"] * 5,
            }
        )
        mock_load_data.return_value = sample_data

        # Create config with temp directory
        data_config = PreprocessingDataConfig(
            raw_data_path=Path("dummy.csv"),
            output_dir=tmp_path / "processed",
            test_size=0.3,
            random_seed=42,
        )
        preprocessing_config = PreprocessingConfig(
            handle_missing=True,
            drop_duplicates=True,
            engineer_features=True,
        )

        # Run pipeline
        pipeline = PreprocessingPipeline(data_config, preprocessing_config)
        x_train, x_test, y_train, y_test = pipeline.run()

        # Verify outputs
        assert len(x_train) > 0
        assert len(x_test) > 0
        assert len(y_train) == len(x_train)
        assert len(y_test) == len(x_test)

        # Verify files were created
        assert (tmp_path / "processed" / "train.csv").exists()
        assert (tmp_path / "processed" / "test.csv").exists()
        assert (tmp_path / "processed" / "preprocessing_metadata.json").exists()

        # Verify metadata
        with open(tmp_path / "processed" / "preprocessing_metadata.json") as f:
            metadata = json.load(f)

        assert "original_columns" in metadata
        assert "columns_after_processing" in metadata
        assert metadata["train_samples"] == len(x_train)
