"""Comprehensive unit tests for model training pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch

from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from entities.configs import TrainingDataConfig, FeatureConfig, ModelConfig
from training.config import load_training_config
from training.training_pipeline import TrainingPipeline


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_dataconfig_initialization_defaults(self):
        """Test DataConfig initialization with default values."""
        config = TrainingDataConfig()

        assert config.train_path == Path("data/processed/train.csv")
        assert config.test_path == Path("data/processed/test.csv")
        assert config.target_column == "y"
        assert config.encode_target is True

    def test_dataconfig_initialization_custom_values(self):
        """Test DataConfig initialization with custom values."""
        config = TrainingDataConfig(
            train_path=Path("custom/train.csv"),
            test_path=Path("custom/test.csv"),
            target_column="target",
            encode_target=False,
        )

        assert config.train_path == Path("custom/train.csv")
        assert config.test_path == Path("custom/test.csv")
        assert config.target_column == "target"
        assert config.encode_target is False

    def test_dataconfig_post_init_converts_strings_to_paths(self):
        """Test that __post_init__ converts string paths to Path objects."""
        config = TrainingDataConfig(
            train_path=Path("data/train.csv"),
            test_path=Path("data/test.csv"),
        )

        assert isinstance(config.train_path, Path)
        assert isinstance(config.test_path, Path)
        assert config.train_path == Path("data/train.csv")
        assert config.test_path == Path("data/test.csv")


class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""

    def test_featureconfig_initialization_defaults(self):
        """Test FeatureConfig initialization with default values."""
        config = FeatureConfig()

        assert config.categorical_features == []
        assert config.numerical_features == []
        assert config.binary_features == []

    def test_featureconfig_initialization_custom_values(self):
        """Test FeatureConfig initialization with custom values."""
        config = FeatureConfig(
            categorical_features=["cat1", "cat2"],
            numerical_features=["num1", "num2", "num3"],
            binary_features=["bin1"],
        )

        assert config.categorical_features == ["cat1", "cat2"]
        assert config.numerical_features == ["num1", "num2", "num3"]
        assert config.binary_features == ["bin1"]


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_modelconfig_initialization_defaults(self):
        """Test ModelConfig initialization with default values."""
        config = ModelConfig()

        assert config.type == "XGBClassifier"
        assert config.parameters == {}

    def test_modelconfig_initialization_custom_values(self):
        """Test ModelConfig initialization with custom values."""
        params = {"learning_rate": 0.1, "max_depth": 5}
        config = ModelConfig(type="XGBClassifier", parameters=params)

        assert config.type == "XGBClassifier"
        assert config.parameters == params

    def test_modelconfig_unsupported_model_type(self):
        """Test that unsupported model type raises ValueError."""
        with pytest.raises(ValueError, match="Model type 'RandomForest' not supported"):
            ModelConfig(type="RandomForest")

    def test_modelconfig_supported_model_type(self):
        """Test that supported model type is accepted."""
        config = ModelConfig(type="XGBClassifier")
        assert config.type == "XGBClassifier"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading configuration from valid YAML file."""
        config_path = tmp_path / "train_config.yaml"
        config_data = {
            "job_name": "test_experiment",
            "data": {
                "train_path": "data/custom/train.csv",
                "test_path": "data/custom/test.csv",
                "target_column": "target",
            },
            "preprocessing": {
                "categorical_features": ["cat1", "cat2"],
                "numerical_features": ["num1", "num2"],
                "binary_features": ["bin1"],
            },
            "model": {
                "type": "XGBClassifier",
                "parameters": {"learning_rate": 0.1, "max_depth": 5},
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        job_name, data_config, feature_config, model_config = load_training_config(config_path)

        # Verify job name
        assert job_name == "test_experiment"

        # Verify DataConfig
        assert data_config.train_path == Path("data/custom/train.csv")
        assert data_config.test_path == Path("data/custom/test.csv")
        assert data_config.target_column == "target"

        # Verify FeatureConfig
        assert feature_config.categorical_features == ["cat1", "cat2"]
        assert feature_config.numerical_features == ["num1", "num2"]
        assert feature_config.binary_features == ["bin1"]

        # Verify ModelConfig
        assert model_config.type == "XGBClassifier"
        assert model_config.parameters == {"learning_rate": 0.1, "max_depth": 5}

    def test_load_config_with_defaults(self, tmp_path):
        """Test loading configuration with missing optional fields uses defaults."""
        config_path = tmp_path / "minimal_config.yaml"
        config_data: Dict[str, Any] = {
            "data": {},
            "preprocessing": {},
            "model": {},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        job_name, data_config, feature_config, model_config = load_training_config(config_path)

        # Verify defaults
        assert job_name == "model_training"
        assert data_config.train_path == Path("data/processed/train.csv")
        assert data_config.test_path == Path("data/processed/test.csv")
        assert data_config.target_column == "y"
        assert feature_config.categorical_features == []
        assert model_config.type == "XGBClassifier"
        assert model_config.parameters == {}

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_training_config("nonexistent_config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test error with malformed YAML."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content:\n  - broken")

        with pytest.raises(yaml.YAMLError, match="Error parsing YAML file"):
            load_training_config(config_path)

    def test_load_config_not_dict(self, tmp_path):
        """Test error when YAML is not a dictionary."""
        config_path = tmp_path / "list_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(["item1", "item2"], f)

        with pytest.raises(ValueError, match="Invalid YAML structure.*expected a dictionary"):
            load_training_config(config_path)

    def test_load_config_data_not_dict(self, tmp_path):
        """Test error when 'data' section is not a dictionary."""
        config_path = tmp_path / "bad_data.yaml"
        config_data = {"data": "not a dict", "preprocessing": {}, "model": {}}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="'data' section in YAML must be a dictionary"):
            load_training_config(config_path)

    def test_load_config_preprocessing_not_dict(self, tmp_path):
        """Test error when 'preprocessing' section is not a dictionary."""
        config_path = tmp_path / "bad_preprocessing.yaml"
        config_data = {"data": {}, "preprocessing": "not a dict", "model": {}}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(
            ValueError, match="'preprocessing' section in YAML must be a dictionary"
        ):
            load_training_config(config_path)

    def test_load_config_model_not_dict(self, tmp_path):
        """Test error when 'model' section is not a dictionary."""
        config_path = tmp_path / "bad_model.yaml"
        config_data = {"data": {}, "preprocessing": {}, "model": ["not", "a", "dict"]}

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="'model' section in YAML must be a dictionary"):
            load_training_config(config_path)

    def test_load_config_unsupported_model_type(self, tmp_path):
        """Test error when model type is not supported."""
        config_path = tmp_path / "unsupported_model.yaml"
        config_data = {
            "data": {},
            "preprocessing": {},
            "model": {"type": "UnsupportedModel"},
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Model type 'UnsupportedModel' not supported"):
            load_training_config(config_path)


class TestTrainingPipelineInit:
    """Tests for TrainingPipeline initialization."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_initialization(self, mock_set_experiment):
        """Test TrainingPipeline initialization."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        assert pipeline.job_name == "test_job"
        assert pipeline.data_config == data_config
        assert pipeline.feature_config == feature_config
        assert pipeline.model_config == model_config
        assert pipeline.pipeline is None
        assert pipeline.test_metrics == {}
        assert pipeline.label_encoder is None

        mock_set_experiment.assert_called_once_with("test_job")


class TestTrainingPipelineLoadData:
    """Tests for TrainingPipeline.load_data method."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_load_data_success(self, mock_set_experiment, tmp_path):
        """Test successful data loading."""
        # Create sample CSV files
        train_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "y": ["no", "yes", "no", "yes", "no"],
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": [6, 7],
                "feature2": [60, 70],
                "y": ["yes", "no"],
            }
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Create pipeline
        data_config = TrainingDataConfig(
            train_path=train_path, test_path=test_path, encode_target=True
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Load data
        x_train, x_test, y_train, y_test = pipeline.load_data()

        # Verify shapes
        assert len(x_train) == 5
        assert len(x_test) == 2
        assert len(y_train) == 5
        assert len(y_test) == 2

        # Verify target column removed from features
        assert "y" not in x_train.columns
        assert "y" not in x_test.columns

        # Verify target encoding
        assert set(y_train.unique()) == {0, 1}
        assert set(y_test.unique()) == {0, 1}
        assert pipeline.label_encoder is not None

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_load_data_without_encoding(self, mock_set_experiment, tmp_path):
        """Test data loading without target encoding."""
        train_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "y": [0, 1, 0],
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": [4, 5],
                "y": [1, 0],
            }
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        data_config = TrainingDataConfig(
            train_path=train_path, test_path=test_path, encode_target=False
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        x_train, x_test, y_train, y_test = pipeline.load_data()

        # Verify no encoding occurred
        assert pipeline.label_encoder is None
        assert y_train.tolist() == [0, 1, 0]
        assert y_test.tolist() == [1, 0]

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_load_data_train_file_not_found(self, mock_set_experiment):
        """Test error when training file doesn't exist."""
        data_config = TrainingDataConfig(
            train_path=Path("nonexistent_train.csv"),
            test_path=Path("nonexistent_test.csv"),
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        with pytest.raises(FileNotFoundError, match="Training data not found"):
            pipeline.load_data()

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_load_data_test_file_not_found(self, mock_set_experiment, tmp_path):
        """Test error when test file doesn't exist."""
        train_df = pd.DataFrame({"feature1": [1, 2], "y": [0, 1]})
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)

        data_config = TrainingDataConfig(
            train_path=train_path,
            test_path=Path("nonexistent_test.csv"),
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        with pytest.raises(FileNotFoundError, match="Test data not found"):
            pipeline.load_data()

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_load_data_missing_target_column(self, mock_set_experiment, tmp_path):
        """Test error when target column is missing."""
        train_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [10, 20, 30]})
        test_df = pd.DataFrame({"feature1": [4, 5], "feature2": [40, 50]})

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        data_config = TrainingDataConfig(
            train_path=train_path, test_path=test_path, target_column="missing_target"
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        with pytest.raises(ValueError, match="Target column 'missing_target' not found"):
            pipeline.load_data()


class TestTrainingPipelinePreprocessing:
    """Tests for TrainingPipeline.create_preprocessing_pipeline method."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_create_preprocessing_pipeline_all_feature_types(self, mock_set_experiment):
        """Test creating preprocessing pipeline with all feature types."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(
            categorical_features=["cat1", "cat2"],
            numerical_features=["num1", "num2", "num3"],
            binary_features=["bin1"],
        )
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        preprocessor = pipeline.create_preprocessing_pipeline()

        # Verify preprocessor structure
        assert len(preprocessor.transformers) == 3
        assert preprocessor.transformers[0][0] == "categorical"
        assert preprocessor.transformers[1][0] == "numerical"
        assert preprocessor.transformers[2][0] == "binary"

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_create_preprocessing_pipeline_only_categorical(self, mock_set_experiment):
        """Test creating preprocessing pipeline with only categorical features."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(categorical_features=["cat1", "cat2"])
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        preprocessor = pipeline.create_preprocessing_pipeline()

        assert len(preprocessor.transformers) == 1
        assert preprocessor.transformers[0][0] == "categorical"

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_create_preprocessing_pipeline_only_numerical(self, mock_set_experiment):
        """Test creating preprocessing pipeline with only numerical features."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(numerical_features=["num1", "num2"])
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        preprocessor = pipeline.create_preprocessing_pipeline()

        assert len(preprocessor.transformers) == 1
        assert preprocessor.transformers[0][0] == "numerical"

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_create_preprocessing_pipeline_empty(self, mock_set_experiment):
        """Test creating preprocessing pipeline with no features."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        preprocessor = pipeline.create_preprocessing_pipeline()

        assert len(preprocessor.transformers) == 0


class TestTrainingPipelineTrainModel:
    """Tests for TrainingPipeline.train_model method."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_train_model_success(self, mock_set_experiment):
        """Test successful model training."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(numerical_features=["feature1", "feature2"])
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"n_estimators": 10, "random_state": 42}
        )

        pipeline_obj = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Create sample data
        x_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5] * 10,
                "feature2": [10, 20, 30, 40, 50] * 10,
            }
        )
        y_train = pd.Series([0, 1, 0, 1, 0] * 10)

        # Train model
        trained_pipeline, training_info = pipeline_obj.train_model(x_train, y_train)

        # Verify pipeline was created
        assert isinstance(trained_pipeline, Pipeline)
        assert pipeline_obj.pipeline is not None
        assert len(trained_pipeline.steps) == 2
        assert trained_pipeline.steps[0][0] == "preprocessor"
        assert trained_pipeline.steps[1][0] == "classifier"

        # Verify training info
        assert training_info["model_type"] == "XGBClassifier"
        assert training_info["n_samples"] == 50
        assert training_info["n_features"] == 2

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_train_model_unsupported_model_type(self, mock_set_experiment):
        """Test error when model type is not supported."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        # Override the validation in ModelConfig for this test
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.type = "UnsupportedModel"
        model_config.parameters = {}

        pipeline_obj = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        x_train = pd.DataFrame({"feature1": [1, 2, 3]})
        y_train = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="Unsupported model type"):
            pipeline_obj.train_model(x_train, y_train)


class TestTrainingPipelineEvaluateModel:
    """Tests for TrainingPipeline.evaluate_model method."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_evaluate_model_success(self, mock_set_experiment):
        """Test successful model evaluation."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(numerical_features=["feature1", "feature2"])
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"n_estimators": 10, "random_state": 42}
        )

        pipeline_obj = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Create and train on sample data
        x_train = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5] * 10,
                "feature2": [10, 20, 30, 40, 50] * 10,
            }
        )
        y_train = pd.Series([0, 1, 0, 1, 0] * 10)
        pipeline_obj.train_model(x_train, y_train)

        # Create test data
        x_test = pd.DataFrame(
            {
                "feature1": [6, 7, 8, 9, 10],
                "feature2": [60, 70, 80, 90, 100],
            }
        )
        y_test = pd.Series([1, 0, 1, 0, 1])

        # Evaluate model
        _, metrics = pipeline_obj.evaluate_model(x_train, y_train, x_test, y_test)

        # Verify metrics are returned
        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1_score" in metrics
        assert "test_roc_auc" in metrics

        # Verify metrics are floats in valid range
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (float, np.floating))
            assert 0 <= metric_value <= 1

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_evaluate_model_before_training(self, mock_set_experiment):
        """Test error when evaluating before training."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline_obj = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        x_test = pd.DataFrame({"feature1": [1, 2, 3]})
        y_test = pd.Series([0, 1, 0])

        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            pipeline_obj.evaluate_model(x_test, y_test, x_test, y_test)


class TestTrainingPipelineLogToMLflow:
    """Tests for TrainingPipeline.log_to_mlflow method."""

    @patch("training.training_pipeline.mlflow.start_run")
    @patch("training.training_pipeline.mlflow.log_params")
    @patch("training.training_pipeline.mlflow.log_param")
    @patch("training.training_pipeline.mlflow.log_metrics")
    @patch("training.training_pipeline.mlflow.sklearn.log_model")
    @patch("training.training_pipeline.mlflow.log_artifact")
    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_log_to_mlflow_success(
        self,
        mock_set_experiment,
        mock_log_artifact,
        mock_log_model,
        mock_log_metrics,
        mock_log_param,
        mock_log_params,
        mock_start_run,
    ):
        """Test successful logging to MLflow."""
        # Set up context manager for mlflow.start_run
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()

        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(
            categorical_features=["cat1"],
            numerical_features=["num1", "num2"],
            binary_features=["bin1"],
        )
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"learning_rate": 0.1, "max_depth": 5}
        )

        pipeline_obj = TrainingPipeline(
            job_name="test_job",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Create and train model
        x_train = pd.DataFrame(
            {
                "cat1": ["a", "b", "c"] * 10,
                "num1": [1, 2, 3] * 10,
                "num2": [10, 20, 30] * 10,
                "bin1": ["yes", "no", "yes"] * 10,
            }
        )
        y_train = pd.Series([0, 1, 0] * 10)
        pipeline_obj.train_model(x_train, y_train)

        # Set some metrics
        pipeline_obj.metrics = {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
            "roc_auc": 0.90,
        }

        # Log to MLflow
        training_info = {
            "model_type": "XGBClassifier",
            "n_samples": 30,
            "n_features": 4,
            "x_train": x_train,
        }
        pipeline_obj.log_to_mlflow(training_info)

        # Verify MLflow calls
        mock_log_params.assert_called_once_with(model_config.parameters)

        # Verify log_param was called with correct arguments
        log_param_calls = mock_log_param.call_args_list
        param_dict = {call[0][0]: call[0][1] for call in log_param_calls}
        assert param_dict["model_type"] == "XGBClassifier"
        assert param_dict["n_train_samples"] == 30
        assert param_dict["n_features"] == 4
        assert param_dict["n_categorical_features"] == 1
        assert param_dict["n_numerical_features"] == 2
        assert param_dict["n_binary_features"] == 1


class TestTrainingPipelineRun:
    """Tests for TrainingPipeline.run method (integration test)."""

    @patch("training.training_pipeline.mlflow.set_experiment")
    @patch("training.training_pipeline.mlflow.start_run")
    @patch("training.training_pipeline.mlflow.log_params")
    @patch("training.training_pipeline.mlflow.log_param")
    @patch("training.training_pipeline.mlflow.log_metrics")
    @patch("training.training_pipeline.mlflow.sklearn.log_model")
    @patch("training.training_pipeline.mlflow.log_artifact")
    def test_run_complete_pipeline(
        self,
        mock_log_artifact,
        mock_log_model,
        mock_log_metrics,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        mock_set_experiment,
        tmp_path,
    ):
        """Test complete pipeline execution."""
        # Set up MLflow mocks
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()

        # Create sample data files
        train_df = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5] * 10,
                "num2": [10, 20, 30, 40, 50] * 10,
                "y": [0, 1, 0, 1, 0] * 10,
            }
        )
        test_df = pd.DataFrame(
            {
                "num1": [6, 7, 8, 9, 10],
                "num2": [60, 70, 80, 90, 100],
                "y": [1, 0, 1, 0, 1],
            }
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Create pipeline
        data_config = TrainingDataConfig(
            train_path=train_path, test_path=test_path, encode_target=False
        )
        feature_config = FeatureConfig(numerical_features=["num1", "num2"])
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"n_estimators": 10, "random_state": 42}
        )

        pipeline = TrainingPipeline(
            job_name="test_complete_pipeline",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Run complete pipeline
        trained_pipeline, metrics = pipeline.run()

        # Verify outputs
        assert isinstance(trained_pipeline, Pipeline)
        assert isinstance(metrics, dict)
        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1_score" in metrics
        assert "test_roc_auc" in metrics

        # Verify MLflow was called
        mock_set_experiment.assert_called()
        mock_start_run.assert_called()

    @patch("training.training_pipeline.mlflow.set_experiment")
    def test_run_with_missing_data_file(self, mock_set_experiment):
        """Test error handling when data file is missing."""
        data_config = TrainingDataConfig(
            train_path=Path("nonexistent_train.csv"),
            test_path=Path("nonexistent_test.csv"),
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()

        pipeline = TrainingPipeline(
            job_name="test_error",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        with pytest.raises(FileNotFoundError):
            pipeline.run()


class TestRunTrainingMain:
    """Tests for run_training.py main function."""

    @patch("src.run_training.TrainingPipeline")
    @patch("src.run_training.load_training_config")
    @patch("src.run_training.Path.mkdir")
    def test_main_success(self, mock_mkdir, mock_load_config, mock_training_pipeline):
        """Test successful execution of main function."""
        # Mock configuration loading
        mock_data_config = TrainingDataConfig()
        mock_feature_config = FeatureConfig()
        mock_model_config = ModelConfig()
        mock_load_config.return_value = (
            "test_experiment",
            mock_data_config,
            mock_feature_config,
            mock_model_config,
        )

        # Mock pipeline execution
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = (
            Mock(),
            {
                "accuracy": 0.85,
                "precision": 0.80,
                "recall": 0.75,
                "f1_score": 0.77,
                "roc_auc": 0.90,
            },
        )
        mock_training_pipeline.return_value = mock_pipeline_instance

        # Import and run main
        from src.run_training import main

        with patch("sys.argv", ["run_training.py", "--config", "test_config.yaml"]):
            exit_code = main()

        # Verify success
        assert exit_code == 0
        mock_load_config.assert_called_once()
        mock_training_pipeline.assert_called_once()
        mock_pipeline_instance.run.assert_called_once()

    @patch("src.run_training.load_training_config")
    def test_main_config_load_error(self, mock_load_config):
        """Test error handling when config loading fails."""
        mock_load_config.side_effect = Exception("Config load error")

        from src.run_training import main

        with patch("sys.argv", ["run_training.py", "--config", "bad_config.yaml"]):
            exit_code = main()

        assert exit_code == 1

    @patch("src.run_training.TrainingPipeline")
    @patch("src.run_training.load_training_config")
    @patch("src.run_training.Path.mkdir")
    def test_main_training_error(self, mock_mkdir, mock_load_config, mock_training_pipeline):
        """Test error handling when training fails."""
        # Mock configuration loading
        mock_data_config = TrainingDataConfig()
        mock_feature_config = FeatureConfig()
        mock_model_config = ModelConfig()
        mock_load_config.return_value = (
            "test_experiment",
            mock_data_config,
            mock_feature_config,
            mock_model_config,
        )

        # Mock pipeline execution failure
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.side_effect = Exception("Training error")
        mock_training_pipeline.return_value = mock_pipeline_instance

        from src.run_training import main

        with patch("sys.argv", ["run_training.py"]):
            exit_code = main()

        assert exit_code == 1

    @patch("src.run_training.TrainingPipeline")
    @patch("src.run_training.load_training_config")
    @patch("src.run_training.Path.mkdir")
    def test_main_default_config_path(self, mock_mkdir, mock_load_config, mock_training_pipeline):
        """Test that default config path is used when not specified."""
        # Mock configuration loading
        mock_data_config = TrainingDataConfig()
        mock_feature_config = FeatureConfig()
        mock_model_config = ModelConfig()
        mock_load_config.return_value = (
            "test_experiment",
            mock_data_config,
            mock_feature_config,
            mock_model_config,
        )

        # Mock pipeline execution
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.run.return_value = (Mock(), {"accuracy": 0.85})
        mock_training_pipeline.return_value = mock_pipeline_instance

        from src.run_training import main

        with patch("sys.argv", ["run_training.py"]):
            exit_code = main()

        # Verify default config path was used
        mock_load_config.assert_called_once_with("confs/training.yaml")
        assert exit_code == 0
