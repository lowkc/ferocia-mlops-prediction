"""Unit tests for data preprocessing pipeline."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from preprocessing.config import DataConfig
from preprocessing.data_preprocessing import (
    DataCleaner,
    DataLoader,
    DataSplitter,
    FeatureEngineer,
    PreprocessingPipeline,
)


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.raw_data_path == Path("data/dataset.csv")
        assert config.output_dir == Path("data/processed")
        assert config.test_size == 0.2
        assert config.random_seed == 42
        assert config.stratify is True
        assert config.log_level == "INFO"
        assert config.save_metadata is True

    def test_invalid_test_size(self):
        """Test validation for invalid test_size."""
        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            DataConfig(test_size=1.5)

        with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
            DataConfig(test_size=0.0)

    def test_invalid_random_seed(self):
        """Test validation for invalid random_seed."""
        with pytest.raises(ValueError, match="random_seed must be non-negative"):
            DataConfig(random_seed=-1)


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

    def test_transform_pdays(self):
        """Test pdays transformation."""
        df = pd.DataFrame({"pdays": [-1, 5, -1, 10, 15]})

        logger = Mock()
        cleaner = DataCleaner(logger)
        df_transformed = cleaner.transform_pdays(df)

        assert "pdays" not in df_transformed.columns
        assert "previous_contact" in df_transformed.columns
        assert "days_since_last_contact" in df_transformed.columns

        # Check values
        assert df_transformed["previous_contact"].tolist() == [0, 1, 0, 1, 1]
        assert df_transformed["days_since_last_contact"].tolist() == [0, 5, 0, 10, 15]

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


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_encode_binary_features(self):
        """Test binary feature encoding."""
        df = pd.DataFrame(
            {"binary1": ["yes", "no", "yes"], "binary2": ["no", "yes", "no"], "other": [1, 2, 3]}
        )

        logger = Mock()
        engineer = FeatureEngineer(logger)
        df_encoded = engineer.encode_binary_features(df, ["binary1", "binary2"])

        assert df_encoded["binary1"].tolist() == [1, 0, 1]
        assert df_encoded["binary2"].tolist() == [0, 1, 0]
        assert df_encoded["other"].tolist() == [1, 2, 3]

    def test_encode_categorical_features(self):
        """Test one-hot encoding of categorical features."""
        df = pd.DataFrame({"cat1": ["a", "b", "a"], "cat2": ["x", "y", "x"], "num": [1, 2, 3]})

        logger = Mock()
        engineer = FeatureEngineer(logger)
        df_encoded = engineer.encode_categorical_features(df, ["cat1", "cat2"])

        # Should have drop_first=True, so only n-1 categories per feature
        assert "cat1_b" in df_encoded.columns
        assert "cat2_y" in df_encoded.columns
        assert "num" in df_encoded.columns
        assert len(df_encoded.columns) == 3  # num + 2 one-hot encoded columns


class TestDataSplitter:
    """Tests for DataSplitter class."""

    def test_split_data(self):
        """Test train/test splitting."""
        df = pd.DataFrame(
            {"feature1": range(100), "feature2": range(100, 200), "target": [0, 1] * 50}
        )

        logger = Mock()
        splitter = DataSplitter(logger)
        X_train, X_test, y_train, y_test = splitter.split_data(
            df, target_column="target", test_size=0.2, random_seed=42, stratify=True
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert "target" not in X_train.columns
        assert "target" not in X_test.columns

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
        config = DataConfig(
            raw_data_path=Path("dummy.csv"),
            output_dir=tmp_path / "processed",
            test_size=0.2,
            random_seed=42,
        )

        # Run pipeline
        pipeline = PreprocessingPipeline(config)
        X_train, X_test, y_train, y_test = pipeline.run()

        # Verify outputs
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        # Verify files were created
        assert (tmp_path / "processed" / "train.csv").exists()
        assert (tmp_path / "processed" / "test.csv").exists()
        assert (tmp_path / "processed" / "preprocessing_metadata.json").exists()

        # Verify metadata
        with open(tmp_path / "processed" / "preprocessing_metadata.json") as f:
            metadata = json.load(f)

        assert "original_columns" in metadata
        assert "processed_columns" in metadata
        assert metadata["train_samples"] == len(X_train)
