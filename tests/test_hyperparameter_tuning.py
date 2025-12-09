"""Comprehensive unit tests for hyperparameter tuning pipeline."""

from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import pytest
import optuna

from src.entities.configs import (
    TrainingDataConfig,
    FeatureConfig,
    ModelConfig,
    TuningConfig,
)
from src.training.hyperparameter_tuning import HyperparameterTuningPipeline


@pytest.fixture
def sample_data():
    """Fixture providing sample training and test data."""
    x_train = pd.DataFrame({"num1": list(range(50)), "num2": list(range(50, 100))})
    y_train = pd.Series([0, 1] * 25)
    x_test = pd.DataFrame({"num1": list(range(10)), "num2": list(range(10, 20))})
    y_test = pd.Series([0, 1] * 5)
    return x_train, x_test, y_train, y_test


@pytest.fixture
def tuning_config():
    """Fixture providing a sample tuning configuration."""
    return TuningConfig(
        n_trials=3,
        direction="maximize",
        random_state=42,
        params={
            "max_depth": {"type": "int", "low": 3, "high": 10, "step": 1},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "n_estimators": 100,  # Fixed parameter
        },
    )


class TestHyperparameterTuningPipelineInit:
    """Tests for HyperparameterTuningPipeline initialization."""

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_initialization(self, mock_set_experiment):
        """Test pipeline initialization."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        model_config = ModelConfig()
        tuning_config = TuningConfig()

        pipeline = HyperparameterTuningPipeline(
            job_name="test_tuning",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        assert pipeline.job_name == "test_tuning_tuning"
        assert pipeline.data_config == data_config
        assert pipeline.feature_config == feature_config
        assert pipeline.model_config == model_config
        assert pipeline.tuning_config == tuning_config
        assert pipeline.x_train is None
        assert pipeline.y_train is None
        assert pipeline.x_test is None
        assert pipeline.y_test is None

        # Should be called twice - once by parent TrainingPipeline, once by HyperparameterTuningPipeline
        assert mock_set_experiment.call_count == 2

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_logger_setup(self, mock_set_experiment):
        """Test that logger is properly configured."""
        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        model_config = ModelConfig()
        tuning_config = TuningConfig()

        pipeline = HyperparameterTuningPipeline(
            job_name="test_tuning",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        assert pipeline.logger is not None
        assert pipeline.logger.name == "hyperparameter_tuning"
        assert len(pipeline.logger.handlers) >= 1  # At least console handler


class TestHyperparameterTuningPipelineLoadData:
    """Tests for load_data method."""

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_load_data_success(self, mock_set_experiment, tmp_path):
        """Test successful data loading."""
        # Create sample CSV files
        train_df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "y": [0, 1, 0, 1, 0],
            }
        )
        test_df = pd.DataFrame(
            {
                "feature1": [6, 7],
                "feature2": [60, 70],
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
        tuning_config = TuningConfig()

        pipeline = HyperparameterTuningPipeline(
            job_name="test_tuning",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        pipeline.load_data()

        assert pipeline.x_train is not None
        assert pipeline.x_test is not None
        assert pipeline.y_train is not None
        assert pipeline.y_test is not None

        assert len(pipeline.x_train) == 5
        assert len(pipeline.x_test) == 2
        assert len(pipeline.y_train) == 5
        assert len(pipeline.y_test) == 2


class TestSuggestHyperparameters:
    """Tests for _suggest_hyperparameters method."""

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_categorical_param(self, mock_set_experiment):
        """Test suggesting categorical parameter."""
        tuning_config = TuningConfig(
            params={"param1": {"type": "categorical", "choices": ["a", "b", "c"]}}
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = "b"

        params = pipeline._suggest_hyperparameters(trial)

        trial.suggest_categorical.assert_called_once_with("param1", ["a", "b", "c"])
        assert params["param1"] == "b"

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_int_param(self, mock_set_experiment):
        """Test suggesting integer parameter."""
        tuning_config = TuningConfig(
            params={"param1": {"type": "int", "low": 1, "high": 10, "step": 2}}
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)
        trial.suggest_int.return_value = 5

        params = pipeline._suggest_hyperparameters(trial)

        trial.suggest_int.assert_called_once_with("param1", 1, 10, step=2)
        assert params["param1"] == 5

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_float_param_linear(self, mock_set_experiment):
        """Test suggesting float parameter with linear scale."""
        tuning_config = TuningConfig(
            params={"param1": {"type": "float", "low": 0.1, "high": 1.0, "log": False}}
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.5

        params = pipeline._suggest_hyperparameters(trial)

        trial.suggest_float.assert_called_once_with("param1", 0.1, 1.0)
        assert params["param1"] == 0.5

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_float_param_log(self, mock_set_experiment):
        """Test suggesting float parameter with log scale."""
        tuning_config = TuningConfig(
            params={"param1": {"type": "float", "low": 0.01, "high": 1.0, "log": True}}
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)
        trial.suggest_float.return_value = 0.1

        params = pipeline._suggest_hyperparameters(trial)

        trial.suggest_float.assert_called_once_with("param1", 0.01, 1.0, log=True)
        assert params["param1"] == 0.1

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_fixed_param(self, mock_set_experiment):
        """Test handling fixed parameter."""
        tuning_config = TuningConfig(params={"param1": 100})

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)

        params = pipeline._suggest_hyperparameters(trial)

        assert params["param1"] == 100
        # No suggest methods should be called for fixed params
        trial.suggest_categorical.assert_not_called()
        trial.suggest_int.assert_not_called()
        trial.suggest_float.assert_not_called()

    @patch("training.hyperparameter_tuning.mlflow.set_experiment")
    def test_suggest_mixed_params(self, mock_set_experiment):
        """Test suggesting mixed parameter types."""
        tuning_config = TuningConfig(
            params={
                "cat_param": {"type": "categorical", "choices": ["a", "b"]},
                "int_param": {"type": "int", "low": 1, "high": 10},
                "float_param": {"type": "float", "low": 0.1, "high": 1.0},
                "fixed_param": 42,
            }
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=tuning_config,
        )

        trial = Mock(spec=optuna.Trial)
        trial.suggest_categorical.return_value = "a"
        trial.suggest_int.return_value = 5
        trial.suggest_float.return_value = 0.5

        params = pipeline._suggest_hyperparameters(trial)

        assert params["cat_param"] == "a"
        assert params["int_param"] == 5
        assert params["float_param"] == 0.5
        assert params["fixed_param"] == 42


class TestObjectiveFunction:
    """Tests for _objective method."""

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    @patch("src.training.hyperparameter_tuning.mlflow.start_run")
    @patch("src.training.hyperparameter_tuning.mlflow.log_params")
    @patch("src.training.hyperparameter_tuning.mlflow.log_param")
    @patch("src.training.hyperparameter_tuning.mlflow.log_metrics")
    def test_objective_runs_cv(
        self,
        mock_log_metrics,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        mock_set_experiment,
        sample_data,
    ):
        """Test that objective function runs cross-validation."""
        x_train, x_test, y_train, y_test = sample_data

        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()

        data_config = TrainingDataConfig()
        feature_config = FeatureConfig(numerical_features=["num1", "num2"])
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"random_state": 42, "n_estimators": 10}
        )
        tuning_config = TuningConfig(
            n_trials=1,
            params={"max_depth": {"type": "int", "low": 3, "high": 5}},
            random_state=42,
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        pipeline.x_train = x_train
        pipeline.y_train = y_train

        trial = Mock(spec=optuna.Trial)
        trial.number = 0
        trial.suggest_int.return_value = 4

        result = pipeline._objective(trial)

        # Verify result is a float
        assert isinstance(result, (float, np.floating))
        assert 0 <= result <= 1

        # Verify MLflow logging was called
        mock_log_params.assert_called_once()
        mock_log_param.assert_called()
        mock_log_metrics.assert_called()

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_objective_unsupported_model_type(self, mock_set_experiment, sample_data):
        """Test error handling for unsupported model type."""
        x_train, x_test, y_train, y_test = sample_data

        data_config = TrainingDataConfig()
        feature_config = FeatureConfig()
        # Create unsupported model config
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.type = "UnsupportedModel"
        model_config.parameters = {}

        tuning_config = TuningConfig()

        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        pipeline.x_train = x_train
        pipeline.y_train = y_train

        trial = Mock(spec=optuna.Trial)
        trial.number = 0

        with patch("src.training.hyperparameter_tuning.mlflow.start_run"):
            with pytest.raises(ValueError, match="Unsupported model type"):
                pipeline._objective(trial)


class TestPlotMethods:
    """Tests for plotting methods."""

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_plot_optimization_history(self, mock_set_experiment):
        """Test optimization history plot creation."""
        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=TuningConfig(),
        )

        study = Mock(spec=optuna.Study)

        with patch("src.training.hyperparameter_tuning.plot_optimization_history") as mock_plot:
            mock_fig = Mock()
            mock_fig.write_html = Mock()
            mock_plot.return_value = mock_fig

            result = pipeline._plot_optimization_history(study)

            assert "optimization_history.html" in result
            mock_plot.assert_called_once_with(study)
            mock_fig.write_html.assert_called_once()

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_plot_param_importances_success(self, mock_set_experiment):
        """Test parameter importance plot creation."""
        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=TuningConfig(),
        )

        study = Mock(spec=optuna.Study)

        with patch("src.training.hyperparameter_tuning.plot_param_importances") as mock_plot:
            mock_fig = Mock()
            mock_fig.write_html = Mock()
            mock_plot.return_value = mock_fig

            result = pipeline._plot_param_importances(study)

            assert result is not None
            assert "param_importances.html" in result
            mock_plot.assert_called_once_with(study)

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_plot_param_importances_failure(self, mock_set_experiment):
        """Test parameter importance plot handles errors."""
        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=TuningConfig(),
        )

        study = Mock(spec=optuna.Study)

        with patch("src.training.hyperparameter_tuning.plot_param_importances") as mock_plot:
            mock_plot.side_effect = Exception("Plot failed")

            result = pipeline._plot_param_importances(study)

            assert result is None

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_plot_param_slice_success(self, mock_set_experiment):
        """Test parameter slice plot creation."""
        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=TuningConfig(),
        )

        study = Mock(spec=optuna.Study)

        with patch("src.training.hyperparameter_tuning.plot_slice") as mock_plot:
            mock_fig = Mock()
            mock_fig.write_html = Mock()
            mock_plot.return_value = mock_fig

            result = pipeline._plot_param_slice(study)

            assert result is not None
            assert "param_slice.html" in result
            mock_plot.assert_called_once_with(study)

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_plot_param_slice_failure(self, mock_set_experiment):
        """Test parameter slice plot handles errors."""
        pipeline = HyperparameterTuningPipeline(
            job_name="test",
            data_config=TrainingDataConfig(),
            feature_config=FeatureConfig(),
            model_config=ModelConfig(),
            tuning_config=TuningConfig(),
        )

        study = Mock(spec=optuna.Study)

        with patch("src.training.hyperparameter_tuning.plot_slice") as mock_plot:
            mock_plot.side_effect = Exception("Plot failed")

            result = pipeline._plot_param_slice(study)

            assert result is None


class TestRunMethod:
    """Tests for run method (integration test)."""

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    @patch("src.training.hyperparameter_tuning.mlflow.start_run")
    @patch("src.training.hyperparameter_tuning.mlflow.log_params")
    @patch("src.training.hyperparameter_tuning.mlflow.log_param")
    @patch("src.training.hyperparameter_tuning.mlflow.log_metrics")
    @patch("src.training.hyperparameter_tuning.mlflow.log_metric")
    @patch("src.training.hyperparameter_tuning.mlflow.log_artifact")
    @patch("src.training.hyperparameter_tuning.log_model_to_mlflow")
    @patch("src.training.hyperparameter_tuning.log_class_distribution")
    @patch("src.training.hyperparameter_tuning.create_and_log_plots")
    def test_run_complete_pipeline(
        self,
        mock_plots,
        mock_log_class_dist,
        mock_log_model,
        mock_log_artifact,
        mock_log_metric,
        mock_log_metrics,
        mock_log_param,
        mock_log_params,
        mock_start_run,
        mock_set_experiment,
        tmp_path,
    ):
        """Test complete tuning pipeline execution."""
        # Set up MLflow mocks
        mock_start_run.return_value.__enter__ = Mock()
        mock_start_run.return_value.__exit__ = Mock()

        # Create sample data files
        train_df = pd.DataFrame(
            {
                "num1": list(range(30)),
                "num2": list(range(30, 60)),
                "y": [0, 1] * 15,
            }
        )
        test_df = pd.DataFrame(
            {
                "num1": list(range(10)),
                "num2": list(range(10, 20)),
                "y": [0, 1] * 5,
            }
        )

        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Create configs
        data_config = TrainingDataConfig(
            train_path=train_path, test_path=test_path, encode_target=False
        )
        feature_config = FeatureConfig(numerical_features=["num1", "num2"])
        model_config = ModelConfig(
            type="XGBClassifier", parameters={"random_state": 42, "n_estimators": 10}
        )
        tuning_config = TuningConfig(
            n_trials=2,
            params={"max_depth": {"type": "int", "low": 3, "high": 5}},
            random_state=42,
        )

        pipeline = HyperparameterTuningPipeline(
            job_name="test_tuning",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        # Run pipeline
        result = pipeline.run()

        # Verify results
        assert "best_params" in result
        assert "best_value" in result
        assert "test_metrics" in result
        assert "study" in result

        # Verify MLflow was called
        assert mock_start_run.call_count >= 1
        assert mock_log_params.call_count >= 1

    @patch("src.training.hyperparameter_tuning.mlflow.set_experiment")
    def test_run_with_missing_data(self, mock_set_experiment):
        """Test error handling when data loading fails."""
        data_config = TrainingDataConfig(
            train_path=Path("nonexistent_train.csv"),
            test_path=Path("nonexistent_test.csv"),
        )
        feature_config = FeatureConfig()
        model_config = ModelConfig()
        tuning_config = TuningConfig()

        pipeline = HyperparameterTuningPipeline(
            job_name="test_error",
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        with pytest.raises(Exception):  # Could be FileNotFoundError or other
            pipeline.run()


class TestTuningConfig:
    """Tests for TuningConfig dataclass."""

    def test_tuningconfig_initialization_defaults(self):
        """Test TuningConfig initialization with defaults."""
        config = TuningConfig()

        assert config.n_trials == 50  # Default is 50 in the actual implementation
        assert config.direction == "maximize"
        assert config.random_state == 42
        assert config.params == {}

    def test_tuningconfig_initialization_custom(self):
        """Test TuningConfig initialization with custom values."""
        params = {
            "param1": {"type": "int", "low": 1, "high": 10},
            "param2": 100,
        }
        config = TuningConfig(
            n_trials=20,
            direction="minimize",
            random_state=123,
            params=params,
        )

        assert config.n_trials == 20
        assert config.direction == "minimize"
        assert config.random_state == 123
        assert config.params == params

    def test_tuningconfig_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be 'maximize' or 'minimize'"):
            TuningConfig(direction="invalid")

    def test_tuningconfig_valid_maximize(self):
        """Test that 'maximize' direction is valid."""
        config = TuningConfig(direction="maximize")
        assert config.direction == "maximize"

    def test_tuningconfig_valid_minimize(self):
        """Test that 'minimize' direction is valid."""
        config = TuningConfig(direction="minimize")
        assert config.direction == "minimize"
