"""Comprehensive unit tests for plotting utilities."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from src.utils.plotting_utils import (
    save_plot,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    create_and_log_plots,
)


@pytest.fixture
def sample_predictions():
    """Fixture providing sample predictions for testing."""
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_pred_proba = np.array([0.2, 0.6, 0.8, 0.9, 0.1, 0.4, 0.85, 0.15])
    return y_true, y_pred, y_pred_proba


@pytest.fixture
def sample_pipeline():
    """Fixture providing a trained XGBoost pipeline for testing."""
    # Create sample data
    X = pd.DataFrame(
        {"cat1": ["a", "b", "c"] * 10, "num1": np.random.rand(30), "num2": np.random.rand(30)}
    )
    y = np.random.randint(0, 2, 30)

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(drop="first", sparse_output=False), ["cat1"]),
            ("numerical", StandardScaler(), ["num1", "num2"]),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(n_estimators=10, random_state=42)),
        ]
    )

    # Train pipeline
    pipeline.fit(X, y)

    return pipeline


class TestSavePlot:
    """Tests for save_plot function."""

    def test_save_plot_creates_directory(self, tmp_path):
        """Test that save_plot creates necessary directories."""
        output_path = tmp_path / "subdir" / "plot.png"

        # Create a simple plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        result = save_plot(str(output_path))

        assert Path(result).exists()
        assert Path(result).parent.exists()

    def test_save_plot_returns_path(self, tmp_path):
        """Test that save_plot returns the correct path."""
        output_path = tmp_path / "test_plot.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        result = save_plot(str(output_path))

        assert result == str(output_path)

    def test_save_plot_closes_figure(self, tmp_path):
        """Test that save_plot closes the figure after saving."""
        output_path = tmp_path / "plot.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        initial_fig_count = len(plt.get_fignums())

        save_plot(str(output_path))

        final_fig_count = len(plt.get_fignums())

        # Figure count should decrease after save_plot
        assert final_fig_count < initial_fig_count

    def test_save_plot_custom_dpi(self, tmp_path):
        """Test that save_plot respects custom DPI setting."""
        output_path = tmp_path / "plot.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            save_plot(str(output_path), dpi=300)
            mock_savefig.assert_called_once()
            args, kwargs = mock_savefig.call_args
            assert kwargs["dpi"] == 300

    def test_save_plot_default_dpi(self, tmp_path):
        """Test that save_plot uses default DPI of 150."""
        output_path = tmp_path / "plot.png"

        plt.figure()
        plt.plot([1, 2, 3], [1, 2, 3])

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            save_plot(str(output_path))
            mock_savefig.assert_called_once()
            args, kwargs = mock_savefig.call_args
            assert kwargs["dpi"] == 150


class TestPlotConfusionMatrix:
    """Tests for plot_confusion_matrix function."""

    def test_plot_confusion_matrix_creates_file(self, tmp_path, sample_predictions):
        """Test that confusion matrix plot is created."""
        y_true, y_pred, _ = sample_predictions
        output_path = tmp_path / "confusion_matrix.png"

        result = plot_confusion_matrix(y_true, y_pred, str(output_path))

        assert Path(result).exists()
        assert result == str(output_path)

    def test_plot_confusion_matrix_perfect_predictions(self, tmp_path):
        """Test confusion matrix with perfect predictions."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        output_path = tmp_path / "cm_perfect.png"

        result = plot_confusion_matrix(y_true, y_pred, str(output_path))

        assert Path(result).exists()

    def test_plot_confusion_matrix_worst_predictions(self, tmp_path):
        """Test confusion matrix with completely wrong predictions."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        output_path = tmp_path / "cm_worst.png"

        result = plot_confusion_matrix(y_true, y_pred, str(output_path))

        assert Path(result).exists()

    def test_plot_confusion_matrix_custom_title(self, tmp_path, sample_predictions):
        """Test confusion matrix with custom title."""
        y_true, y_pred, _ = sample_predictions
        output_path = tmp_path / "cm_custom.png"

        with patch("src.utils.plotting_utils.plt.title") as mock_title:
            plot_confusion_matrix(y_true, y_pred, str(output_path), title="Custom Title")
            mock_title.assert_called_once_with("Custom Title")

    def test_plot_confusion_matrix_default_title(self, tmp_path, sample_predictions):
        """Test confusion matrix with default title."""
        y_true, y_pred, _ = sample_predictions
        output_path = tmp_path / "cm_default.png"

        with patch("src.utils.plotting_utils.plt.title") as mock_title:
            plot_confusion_matrix(y_true, y_pred, str(output_path))
            mock_title.assert_called_once_with("Confusion Matrix")

    def test_plot_confusion_matrix_default_output_path(self, tmp_path):
        """Test confusion matrix with default output path."""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        # Change to temp directory
        import os

        original_dir = os.getcwd()
        os.chdir(tmp_path)

        try:
            result = plot_confusion_matrix(y_true, y_pred)
            # Default path should be outputs/confusion_matrix.png
            assert "confusion_matrix.png" in result
        finally:
            os.chdir(original_dir)

    @patch("src.utils.plotting_utils.confusion_matrix")
    def test_plot_confusion_matrix_calls_sklearn(self, mock_cm, tmp_path, sample_predictions):
        """Test that sklearn's confusion_matrix is called correctly."""
        y_true, y_pred, _ = sample_predictions
        output_path = tmp_path / "cm.png"

        mock_cm.return_value = np.array([[3, 1], [2, 2]])

        plot_confusion_matrix(y_true, y_pred, str(output_path))

        mock_cm.assert_called_once()
        # Verify y_true and y_pred were passed
        args = mock_cm.call_args[0]
        np.testing.assert_array_equal(args[0], y_true)
        np.testing.assert_array_equal(args[1], y_pred)


class TestPlotROCCurve:
    """Tests for plot_roc_curve function."""

    def test_plot_roc_curve_creates_file(self, tmp_path, sample_predictions):
        """Test that ROC curve plot is created."""
        y_true, _, y_pred_proba = sample_predictions
        output_path = tmp_path / "roc_curve.png"

        result = plot_roc_curve(y_true, y_pred_proba, str(output_path))

        assert Path(result).exists()
        assert result == str(output_path)

    def test_plot_roc_curve_perfect_predictions(self, tmp_path):
        """Test ROC curve with perfect predictions."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.9, 0.95])
        output_path = tmp_path / "roc_perfect.png"

        result = plot_roc_curve(y_true, y_pred_proba, str(output_path))

        assert Path(result).exists()

    def test_plot_roc_curve_custom_title(self, tmp_path, sample_predictions):
        """Test ROC curve with custom title."""
        y_true, _, y_pred_proba = sample_predictions
        output_path = tmp_path / "roc_custom.png"

        with patch("src.utils.plotting_utils.plt.title") as mock_title:
            plot_roc_curve(y_true, y_pred_proba, str(output_path), title="Custom ROC")
            mock_title.assert_called_once_with("Custom ROC")

    def test_plot_roc_curve_default_title(self, tmp_path, sample_predictions):
        """Test ROC curve with default title."""
        y_true, _, y_pred_proba = sample_predictions
        output_path = tmp_path / "roc_default.png"

        with patch("src.utils.plotting_utils.plt.title") as mock_title:
            plot_roc_curve(y_true, y_pred_proba, str(output_path))
            mock_title.assert_called_once_with("ROC Curve")

    @patch("src.utils.plotting_utils.roc_curve")
    @patch("src.utils.plotting_utils.roc_auc_score")
    def test_plot_roc_curve_calls_sklearn(
        self, mock_roc_auc, mock_roc_curve, tmp_path, sample_predictions
    ):
        """Test that sklearn functions are called correctly."""
        y_true, _, y_pred_proba = sample_predictions
        output_path = tmp_path / "roc.png"

        mock_roc_curve.return_value = (
            np.array([0, 0.5, 1]),
            np.array([0, 0.5, 1]),
            np.array([0.9, 0.5, 0.1]),
        )
        mock_roc_auc.return_value = 0.85

        plot_roc_curve(y_true, y_pred_proba, str(output_path))

        mock_roc_curve.assert_called_once()
        mock_roc_auc.assert_called_once()


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance function."""

    def test_plot_feature_importance_creates_file(self, tmp_path, sample_pipeline):
        """Test that feature importance plot is created."""
        output_path = tmp_path / "feature_importance.png"

        result = plot_feature_importance(sample_pipeline, str(output_path))

        assert Path(result).exists()
        assert result == str(output_path)

    def test_plot_feature_importance_custom_top_n(self, tmp_path, sample_pipeline):
        """Test feature importance with custom top_n parameter."""
        output_path = tmp_path / "feature_importance_top5.png"

        result = plot_feature_importance(sample_pipeline, str(output_path), top_n=5)

        assert Path(result).exists()

    def test_plot_feature_importance_default_top_n(self, tmp_path, sample_pipeline):
        """Test feature importance with default top_n parameter."""
        output_path = tmp_path / "feature_importance_default.png"

        result = plot_feature_importance(sample_pipeline, str(output_path))

        assert Path(result).exists()

    def test_plot_feature_importance_extracts_model(self, tmp_path, sample_pipeline):
        """Test that the classifier is correctly extracted from pipeline."""
        output_path = tmp_path / "fi.png"

        # The function should not raise an error
        result = plot_feature_importance(sample_pipeline, str(output_path))

        assert Path(result).exists()

    def test_plot_feature_importance_gets_feature_names(self, tmp_path):
        """Test that feature names are correctly extracted."""
        # Create a simple pipeline with known features
        X = pd.DataFrame({"num1": [1, 2, 3, 4, 5], "num2": [10, 20, 30, 40, 50]})
        y = np.array([0, 1, 0, 1, 0])

        preprocessor = ColumnTransformer(
            transformers=[("numerical", StandardScaler(), ["num1", "num2"])]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(n_estimators=5, random_state=42)),
            ]
        )

        pipeline.fit(X, y)
        output_path = tmp_path / "fi.png"

        result = plot_feature_importance(pipeline, str(output_path), top_n=2)

        assert Path(result).exists()

    def test_plot_feature_importance_handles_onehot_encoding(self, tmp_path):
        """Test that OneHotEncoder feature names are handled correctly."""
        X = pd.DataFrame({"cat1": ["a", "b", "c", "a", "b"], "num1": [1, 2, 3, 4, 5]})
        y = np.array([0, 1, 0, 1, 0])

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(drop="first", sparse_output=False), ["cat1"]),
                ("numerical", StandardScaler(), ["num1"]),
            ]
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", XGBClassifier(n_estimators=5, random_state=42)),
            ]
        )

        pipeline.fit(X, y)
        output_path = tmp_path / "fi.png"

        result = plot_feature_importance(pipeline, str(output_path))

        assert Path(result).exists()


class TestCreateAndLogPlots:
    """Tests for create_and_log_plots function."""

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_all_success(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test successful creation and logging of all plots."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.return_value = str(tmp_path / "cm.png")
        mock_roc.return_value = str(tmp_path / "roc.png")
        mock_fi.return_value = str(tmp_path / "fi.png")

        result = create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(tmp_path))

        # Verify all plots were created
        assert "confusion_matrix" in result
        assert "roc_curve" in result
        assert "feature_importance" in result

        # Verify MLflow logging
        assert mock_log_artifact.call_count == 3

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_without_mlflow(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test plot creation without MLflow logging."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.return_value = str(tmp_path / "cm.png")
        mock_roc.return_value = str(tmp_path / "roc.png")
        mock_fi.return_value = str(tmp_path / "fi.png")

        result = create_and_log_plots(
            y_true,
            y_pred,
            y_pred_proba,
            sample_pipeline,
            str(tmp_path),
            log_to_mlflow=False,
        )

        # Verify plots were created but not logged
        assert len(result) == 3
        mock_log_artifact.assert_not_called()

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_handles_cm_failure(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that function continues when confusion matrix fails."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.side_effect = Exception("CM failed")
        mock_roc.return_value = str(tmp_path / "roc.png")
        mock_fi.return_value = str(tmp_path / "fi.png")

        result = create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(tmp_path))

        # Should still create other plots
        assert "confusion_matrix" not in result
        assert "roc_curve" in result
        assert "feature_importance" in result

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_handles_roc_failure(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that function continues when ROC curve fails."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.return_value = str(tmp_path / "cm.png")
        mock_roc.side_effect = Exception("ROC failed")
        mock_fi.return_value = str(tmp_path / "fi.png")

        result = create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(tmp_path))

        # Should still create other plots
        assert "confusion_matrix" in result
        assert "roc_curve" not in result
        assert "feature_importance" in result

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_handles_fi_failure(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that function continues when feature importance fails."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.return_value = str(tmp_path / "cm.png")
        mock_roc.return_value = str(tmp_path / "roc.png")
        mock_fi.side_effect = Exception("FI failed")

        result = create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(tmp_path))

        # Should still create other plots
        assert "confusion_matrix" in result
        assert "roc_curve" in result
        assert "feature_importance" not in result

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_all_failures(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that function handles all plots failing gracefully."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.side_effect = Exception("CM failed")
        mock_roc.side_effect = Exception("ROC failed")
        mock_fi.side_effect = Exception("FI failed")

        result = create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(tmp_path))

        # Should return empty dict but not crash
        assert len(result) == 0
        mock_log_artifact.assert_not_called()

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_custom_output_dir(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        tmp_path,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that custom output directory is used."""
        y_true, y_pred, y_pred_proba = sample_predictions
        custom_dir = tmp_path / "custom_output"

        mock_cm.return_value = str(custom_dir / "cm.png")
        mock_roc.return_value = str(custom_dir / "roc.png")
        mock_fi.return_value = str(custom_dir / "fi.png")

        create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline, str(custom_dir))

        # Verify custom paths were used
        cm_call_args = mock_cm.call_args[1]
        assert str(custom_dir) in cm_call_args["output_path"]

    @patch("src.utils.plotting_utils.mlflow.log_artifact")
    @patch("src.utils.plotting_utils.plot_confusion_matrix")
    @patch("src.utils.plotting_utils.plot_roc_curve")
    @patch("src.utils.plotting_utils.plot_feature_importance")
    def test_create_and_log_plots_default_output_dir(
        self,
        mock_fi,
        mock_roc,
        mock_cm,
        mock_log_artifact,
        sample_predictions,
        sample_pipeline,
    ):
        """Test that default output directory is used."""
        y_true, y_pred, y_pred_proba = sample_predictions

        mock_cm.return_value = "outputs/cm.png"
        mock_roc.return_value = "outputs/roc.png"
        mock_fi.return_value = "outputs/fi.png"

        create_and_log_plots(y_true, y_pred, y_pred_proba, sample_pipeline)

        # Verify default "outputs" directory was used
        cm_call_args = mock_cm.call_args[1]
        assert "outputs" in cm_call_args["output_path"]
