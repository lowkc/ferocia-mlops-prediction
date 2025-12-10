"""Comprehensive unit tests for ModelPredictor class."""

from pathlib import Path
from unittest.mock import Mock, patch
import logging

import joblib
import numpy as np
import pandas as pd
import pytest

from src.serving.predictor import ModelPredictor


class TestModelPredictorInitialization:
    """Tests for ModelPredictor initialization."""

    def test_initialization_with_valid_path(self):
        """Test ModelPredictor initialization with valid path."""
        model_path = Path("models/test_model.pkl")
        predictor = ModelPredictor(model_path)

        assert predictor.model_path == model_path
        assert predictor.model is None
        assert isinstance(predictor.logger, logging.Logger)
        assert predictor.logger.name == "model_predictor"

    def test_initialization_with_string_path(self):
        """Test ModelPredictor initialization converts string to Path."""
        model_path = Path("models/test_model.pkl")
        predictor = ModelPredictor(model_path)

        assert isinstance(predictor.model_path, Path)
        assert predictor.model_path == Path("models/test_model.pkl")

    def test_logger_setup(self):
        """Test logger is properly configured."""
        predictor = ModelPredictor(Path("models/test.pkl"))

        assert predictor.logger.level == logging.INFO
        assert len(predictor.logger.handlers) > 0


class TestModelPredictorLoadModel:
    """Tests for ModelPredictor.load_model method."""

    @patch("src.serving.predictor.joblib.load")
    def test_load_model_success(self, mock_joblib_load, tmp_path):
        """Test successful model loading."""
        # Create a mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([0]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.7, 0.3]]))
        mock_joblib_load.return_value = mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()  # Create empty file

        # Load the model
        predictor = ModelPredictor(model_path)
        predictor.load_model()

        assert predictor.model is not None
        assert predictor.model == mock_model
        mock_joblib_load.assert_called_once_with(model_path)

    def test_load_model_file_not_found(self):
        """Test FileNotFoundError when model file doesn't exist."""
        model_path = Path("nonexistent/model.pkl")
        predictor = ModelPredictor(model_path)

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            predictor.load_model()

    def test_load_model_corrupted_file(self, tmp_path):
        """Test exception handling for corrupted model file."""
        # Create a corrupted file
        model_path = tmp_path / "corrupted_model.pkl"
        with open(model_path, "w") as f:
            f.write("This is not a valid pickle file")

        predictor = ModelPredictor(model_path)

        with pytest.raises(Exception):
            predictor.load_model()

    @patch("src.serving.predictor.joblib.load")
    def test_load_model_logs_info(self, mock_joblib_load, tmp_path):
        """Test that load_model logs appropriate messages."""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()  # Create empty file

        predictor = ModelPredictor(model_path)

        with patch.object(predictor.logger, "info") as mock_info:
            predictor.load_model()

            # Verify logging calls
            assert mock_info.call_count == 2
            assert "Loading model from" in mock_info.call_args_list[0][0][0]
            assert "Model loaded successfully" in mock_info.call_args_list[1][0][0]

    def test_load_model_logs_error_on_failure(self):
        """Test that load_model logs error on failure."""
        model_path = Path("nonexistent/model.pkl")
        predictor = ModelPredictor(model_path)

        with patch.object(predictor.logger, "error") as mock_error:
            with pytest.raises(FileNotFoundError):
                predictor.load_model()

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Failed to load model" in mock_error.call_args[0][0]


class TestModelPredictorPredict:
    """Tests for ModelPredictor.predict method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock model
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.array([1]))
        self.mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))

    @patch("src.serving.predictor.joblib.load")
    def test_predict_success(self, mock_joblib_load, tmp_path):
        """Test successful prediction with valid input."""
        mock_joblib_load.return_value = self.mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        # Load model and predict
        predictor = ModelPredictor(model_path)
        predictor.load_model()

        input_data = {
            "age": 35,
            "balance": 1000,
            "day": 15,
            "duration": 200,
            "campaign": 2,
            "previous": 0,
            "total_contacts": 2,
            "days_since_last_contact": 0,
            "previous_contact": 0,
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "job": "technician",
            "marital": "married",
            "education": "secondary",
            "contact": "cellular",
            "month": "may",
            "poutcome": "unknown",
        }

        result = predictor.predict(input_data)

        # Verify result structure
        assert "prediction" in result
        assert "probability" in result
        assert "probabilities" in result
        assert isinstance(result["prediction"], int)
        assert isinstance(result["probability"], float)
        assert isinstance(result["probabilities"], dict)
        assert "class_0" in result["probabilities"]
        assert "class_1" in result["probabilities"]

    @patch("src.serving.predictor.joblib.load")
    def test_predict_returns_correct_values(self, mock_joblib_load, tmp_path):
        """Test that predict returns correct prediction values."""
        mock_joblib_load.return_value = self.mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        input_data = {"feature1": 1, "feature2": 2}
        result = predictor.predict(input_data)

        assert result["prediction"] == 1
        assert result["probability"] == 0.7
        assert result["probabilities"]["class_0"] == 0.3
        assert result["probabilities"]["class_1"] == 0.7

    def test_predict_model_not_loaded(self):
        """Test ValueError when trying to predict without loading model."""
        predictor = ModelPredictor(Path("models/test.pkl"))

        with pytest.raises(ValueError, match="Model must be loaded before making predictions"):
            predictor.predict({"feature1": 1})

    @patch("src.serving.predictor.joblib.load")
    def test_predict_converts_input_to_dataframe(self, mock_joblib_load, tmp_path):
        """Test that predict converts input dict to DataFrame."""
        mock_joblib_load.return_value = self.mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        input_data = {"feature1": 1, "feature2": 2, "feature3": 3}

        with patch("src.serving.predictor.pd.DataFrame") as mock_df:
            mock_df.return_value = pd.DataFrame([input_data])
            predictor.predict(input_data)

            # Verify DataFrame was created with input data (may be called twice due to logging)
            assert mock_df.call_count >= 1
            mock_df.assert_any_call([input_data])

    @patch("src.serving.predictor.joblib.load")
    def test_predict_handles_prediction_failure(self, mock_joblib_load, tmp_path):
        """Test exception handling when prediction fails."""
        # Create a model that raises exception
        bad_model = Mock()
        bad_model.predict = Mock(side_effect=Exception("Prediction error"))
        mock_joblib_load.return_value = bad_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        with pytest.raises(Exception, match="Prediction error"):
            predictor.predict({"feature1": 1})

    @patch("src.serving.predictor.joblib.load")
    def test_predict_logs_info(self, mock_joblib_load, tmp_path):
        """Test that predict logs appropriate messages."""
        mock_joblib_load.return_value = self.mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        with patch.object(predictor.logger, "info") as mock_info:
            predictor.predict({"feature1": 1})

            # Verify logging calls
            assert mock_info.call_count >= 2
            log_messages = [call[0][0] for call in mock_info.call_args_list]
            assert any("Making prediction" in msg for msg in log_messages)
            assert any("Prediction:" in msg for msg in log_messages)

    @patch("src.serving.predictor.joblib.load")
    def test_predict_logs_error_on_failure(self, mock_joblib_load, tmp_path):
        """Test that predict logs error on failure."""
        bad_model = Mock()
        bad_model.predict = Mock(side_effect=Exception("Test error"))
        mock_joblib_load.return_value = bad_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        with patch.object(predictor.logger, "error") as mock_error:
            with pytest.raises(Exception):
                predictor.predict({"feature1": 1})

            # Verify error was logged
            mock_error.assert_called_once()
            assert "Prediction failed" in mock_error.call_args[0][0]

    @pytest.mark.parametrize(
        "prediction,probabilities,expected",
        [
            (0, [0.8, 0.2], {"prediction": 0, "probability": 0.2}),
            (1, [0.3, 0.7], {"prediction": 1, "probability": 0.7}),
            (0, [0.95, 0.05], {"prediction": 0, "probability": 0.05}),
        ],
    )
    @patch("src.serving.predictor.joblib.load")
    def test_predict_various_outputs(
        self, mock_joblib_load, tmp_path, prediction, probabilities, expected
    ):
        """Test predict with various model outputs."""
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([prediction]))
        mock_model.predict_proba = Mock(return_value=np.array([probabilities]))
        mock_joblib_load.return_value = mock_model

        model_path = tmp_path / "model.pkl"
        model_path.touch()

        predictor = ModelPredictor(model_path)
        predictor.load_model()

        result = predictor.predict({"feature1": 1})

        assert result["prediction"] == expected["prediction"]
        assert result["probability"] == expected["probability"]


class TestModelPredictorIntegration:
    """Integration tests for ModelPredictor."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: initialize, load, predict."""
        # Create a simple mock model
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        # Create a simple pipeline
        pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression())])

        # Create sample training data
        X_train = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [10, 20, 30, 40]})
        y_train = [0, 0, 1, 1]

        # Train the model
        pipeline.fit(X_train, y_train)

        # Save the model
        model_path = tmp_path / "trained_model.pkl"
        joblib.dump(pipeline, model_path)

        # Test the predictor
        predictor = ModelPredictor(model_path)
        predictor.load_model()

        # Make a prediction
        input_data = {"feature1": 2.5, "feature2": 25}
        result = predictor.predict(input_data)

        # Verify result structure
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "probability" in result
        assert "probabilities" in result
        assert result["prediction"] in [0, 1]
        assert 0 <= result["probability"] <= 1
        assert 0 <= result["probabilities"]["class_0"] <= 1
        assert 0 <= result["probabilities"]["class_1"] <= 1
        assert (
            abs(result["probabilities"]["class_0"] + result["probabilities"]["class_1"] - 1.0)
            < 0.01
        )
