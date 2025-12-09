"""Comprehensive unit tests for model utilities."""

from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils.model_utils import (
    calculate_metrics,
    log_model_to_mlflow,
    log_metrics_to_mlflow,
    log_class_distribution,
)


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics calculation with perfect predictions."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_calculate_metrics_worst_predictions(self):
        """Test metrics calculation with worst predictions."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.1, 0.2])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0
        assert metrics["roc_auc"] == 0.0

    def test_calculate_metrics_mixed_predictions(self):
        """Test metrics calculation with mixed predictions."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_pred_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.1, 0.4, 0.85, 0.15])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        # Verify metrics are in valid range
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

        # Calculate expected accuracy manually
        expected_accuracy = 6 / 8  # 6 correct out of 8
        assert metrics["accuracy"] == expected_accuracy

    def test_calculate_metrics_with_prefix(self):
        """Test metrics calculation with prefix."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba, prefix="train")

        assert "train_accuracy" in metrics
        assert "train_precision" in metrics
        assert "train_recall" in metrics
        assert "train_f1_score" in metrics
        assert "train_roc_auc" in metrics
        assert "accuracy" not in metrics

    def test_calculate_metrics_no_prefix(self):
        """Test metrics calculation without prefix."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "train_accuracy" not in metrics

    def test_calculate_metrics_all_negative_predictions(self):
        """Test metrics when all predictions are negative."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 0.5
        assert metrics["precision"] == 0.0  # zero_division=0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0

    def test_calculate_metrics_all_positive_predictions(self):
        """Test metrics when all predictions are positive."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_pred_proba = np.array([0.6, 0.7, 0.8, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert metrics["accuracy"] == 0.5
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 1.0
        assert 0 < metrics["f1_score"] < 1

    def test_calculate_metrics_imbalanced_data(self):
        """Test metrics with imbalanced dataset."""
        y_true = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.2, 0.6, 0.8, 0.9])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_calculate_metrics_numpy_array_input(self):
        """Test that function works with numpy arrays for y_true."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = calculate_metrics(pd.Series(y_true), y_pred, y_pred_proba)

        assert metrics["accuracy"] == 1.0

    def test_calculate_metrics_empty_prefix(self):
        """Test metrics calculation with empty string prefix."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = calculate_metrics(y_true, y_pred, y_pred_proba, prefix="")

        # Empty prefix evaluates to falsy, so no prefix is added
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics
        assert "_accuracy" not in metrics


class TestLogModelToMLflow:
    """Tests for log_model_to_mlflow function."""

    @patch("utils.model_utils.mlflow.sklearn.log_model")
    @patch("utils.model_utils.infer_signature")
    def test_log_model_with_signature(self, mock_infer_signature, mock_log_model):
        """Test logging model with signature inference."""
        # Create a simple pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])

        x_train = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})

        # Train the pipeline
        y_train = np.array([0, 1, 0, 1, 0])
        pipeline.fit(x_train, y_train)

        # Create a proper mock signature that can be serialized
        mock_signature = MagicMock()
        mock_signature.to_dict.return_value = {"inputs": [], "outputs": []}
        mock_infer_signature.return_value = mock_signature

        log_model_to_mlflow(
            pipeline=pipeline,
            model_name="test_model",
            x_train=x_train,
            registered_model_name="test_registered",
        )

        mock_infer_signature.assert_called_once()
        mock_log_model.assert_called_once_with(
            pipeline,
            name="test_model",
            registered_model_name="test_registered",
            signature=mock_signature,
        )

    @patch("utils.model_utils.mlflow.sklearn.log_model")
    def test_log_model_without_signature(self, mock_log_model):
        """Test logging model without signature inference."""
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])

        log_model_to_mlflow(pipeline=pipeline, model_name="test_model")

        mock_log_model.assert_called_once_with(
            pipeline, name="test_model", registered_model_name=None, signature=None
        )

    @patch("utils.model_utils.mlflow.sklearn.log_model")
    @patch("utils.model_utils.infer_signature")
    def test_log_model_with_empty_dataframe(self, mock_infer_signature, mock_log_model):
        """Test logging model with empty training dataframe."""
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])

        x_train = pd.DataFrame()

        log_model_to_mlflow(pipeline=pipeline, model_name="test_model", x_train=x_train)

        # Signature should not be inferred for empty dataframe
        mock_infer_signature.assert_not_called()
        mock_log_model.assert_called_once_with(
            pipeline, name="test_model", registered_model_name=None, signature=None
        )

    @patch("utils.model_utils.mlflow.sklearn.log_model")
    @patch("utils.model_utils.infer_signature")
    def test_log_model_with_none_x_train(self, mock_infer_signature, mock_log_model):
        """Test logging model with None x_train."""
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])

        log_model_to_mlflow(pipeline=pipeline, model_name="test_model", x_train=None)

        mock_infer_signature.assert_not_called()
        mock_log_model.assert_called_once_with(
            pipeline, name="test_model", registered_model_name=None, signature=None
        )


class TestLogMetricsToMLflow:
    """Tests for log_metrics_to_mlflow function."""

    @patch("utils.model_utils.mlflow.log_metrics")
    def test_log_metrics_without_prefix(self, mock_log_metrics):
        """Test logging metrics without prefix."""
        metrics = {"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "f1_score": 0.77}

        log_metrics_to_mlflow(metrics)

        mock_log_metrics.assert_called_once_with(metrics)

    @patch("utils.model_utils.mlflow.log_metrics")
    def test_log_metrics_with_prefix(self, mock_log_metrics):
        """Test logging metrics with prefix."""
        metrics = {"accuracy": 0.85, "precision": 0.80, "recall": 0.75}

        log_metrics_to_mlflow(metrics, prefix="test_")

        expected_metrics = {"test_accuracy": 0.85, "test_precision": 0.80, "test_recall": 0.75}
        mock_log_metrics.assert_called_once_with(expected_metrics)

    @patch("utils.model_utils.mlflow.log_metrics")
    def test_log_metrics_empty_dict(self, mock_log_metrics):
        """Test logging empty metrics dictionary."""
        metrics = {}

        log_metrics_to_mlflow(metrics)

        mock_log_metrics.assert_called_once_with({})

    @patch("utils.model_utils.mlflow.log_metrics")
    def test_log_metrics_with_various_values(self, mock_log_metrics):
        """Test logging metrics with various numeric values."""
        metrics = {"metric1": 0.0, "metric2": 1.0, "metric3": 0.5555555, "metric4": 100}

        log_metrics_to_mlflow(metrics, prefix="val_")

        expected_metrics = {
            "val_metric1": 0.0,
            "val_metric2": 1.0,
            "val_metric3": 0.5555555,
            "val_metric4": 100,
        }
        mock_log_metrics.assert_called_once_with(expected_metrics)


class TestLogClassDistribution:
    """Tests for log_class_distribution function."""

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_balanced(self, mock_log_param):
        """Test logging class distribution for balanced dataset."""
        y_train = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
        y_test = pd.Series([0, 1, 0, 1])

        log_class_distribution(y_train, y_test)

        # Check that log_param was called with correct arguments
        expected_calls = [
            call("train_class_0_count", 4),
            call("train_class_1_count", 4),
            call("train_class_0_pct", 50.0),
            call("train_class_1_pct", 50.0),
            call("test_class_0_count", 2),
            call("test_class_1_count", 2),
            call("test_class_0_pct", 50.0),
            call("test_class_1_pct", 50.0),
        ]

        assert mock_log_param.call_count == 8
        mock_log_param.assert_has_calls(expected_calls, any_order=True)

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_imbalanced(self, mock_log_param):
        """Test logging class distribution for imbalanced dataset."""
        y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # 70% class 0, 30% class 1
        y_test = pd.Series([0, 0, 0, 1])  # 75% class 0, 25% class 1

        log_class_distribution(y_train, y_test)

        # Verify percentages
        calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}

        assert calls["train_class_0_count"] == 7
        assert calls["train_class_1_count"] == 3
        assert calls["train_class_0_pct"] == 70.0
        assert calls["train_class_1_pct"] == 30.0
        assert calls["test_class_0_count"] == 3
        assert calls["test_class_1_count"] == 1
        assert calls["test_class_0_pct"] == 75.0
        assert calls["test_class_1_pct"] == 25.0

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_all_one_class_train(self, mock_log_param):
        """Test logging when training set has only one class."""
        y_train = pd.Series([0, 0, 0, 0])
        y_test = pd.Series([0, 1])

        log_class_distribution(y_train, y_test)

        calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}

        assert calls["train_class_0_count"] == 4
        assert calls["train_class_1_count"] == 0
        assert calls["train_class_0_pct"] == 100.0
        assert calls["train_class_1_pct"] == 0.0

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_all_one_class_test(self, mock_log_param):
        """Test logging when test set has only one class."""
        y_train = pd.Series([0, 1, 0, 1])
        y_test = pd.Series([1, 1, 1])

        log_class_distribution(y_train, y_test)

        calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}

        assert calls["test_class_0_count"] == 0
        assert calls["test_class_1_count"] == 3
        assert calls["test_class_0_pct"] == 0.0
        assert calls["test_class_1_pct"] == 100.0

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_single_sample(self, mock_log_param):
        """Test logging with single sample datasets."""
        y_train = pd.Series([0])
        y_test = pd.Series([1])

        log_class_distribution(y_train, y_test)

        calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}

        assert calls["train_class_0_count"] == 1
        assert calls["train_class_1_count"] == 0
        assert calls["train_class_0_pct"] == 100.0
        assert calls["train_class_1_pct"] == 0.0
        assert calls["test_class_0_count"] == 0
        assert calls["test_class_1_count"] == 1
        assert calls["test_class_0_pct"] == 0.0
        assert calls["test_class_1_pct"] == 100.0

    @patch("utils.model_utils.mlflow.log_param")
    def test_log_class_distribution_large_dataset(self, mock_log_param):
        """Test logging with large dataset."""
        # Create large imbalanced dataset
        y_train = pd.Series([0] * 900 + [1] * 100)  # 90% class 0
        y_test = pd.Series([0] * 80 + [1] * 20)  # 80% class 0

        log_class_distribution(y_train, y_test)

        calls = {call[0][0]: call[0][1] for call in mock_log_param.call_args_list}

        assert calls["train_class_0_count"] == 900
        assert calls["train_class_1_count"] == 100
        assert calls["train_class_0_pct"] == 90.0
        assert calls["train_class_1_pct"] == 10.0
        assert calls["test_class_0_count"] == 80
        assert calls["test_class_1_count"] == 20
        assert calls["test_class_0_pct"] == 80.0
        assert calls["test_class_1_pct"] == 20.0
