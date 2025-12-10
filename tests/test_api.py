"""Comprehensive unit tests for FastAPI serving application."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from src.serving.api import (
    app,
    PredictionInput,
    PredictionOutput,
    HealthResponse,
    ModelInfoResponse,
    load_config,
    initialize_predictor,
)


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration file."""
    config_data = {
        "model": {
            "name": "test_model",
            "local_storage_path": str(tmp_path / "models"),
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "workers": 1,
            "log_level": "info",
        },
    }
    config_path = tmp_path / "deployment.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path, config_data


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestPydanticModels:
    """Tests for Pydantic input/output models."""

    def test_prediction_input_valid_data(self):
        """Test PredictionInput with valid data."""
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

        prediction_input = PredictionInput(**input_data)  # type: ignore[arg-type]

        assert prediction_input.age == 35
        assert prediction_input.balance == 1000
        assert prediction_input.job == "technician"

    @pytest.mark.parametrize(
        "invalid_field,invalid_value,error_match",
        [
            ("age", 17, "greater than or equal to 18"),  # Age too low
            ("age", 101, "less than or equal to 100"),  # Age too high
            ("day", 0, "greater than or equal to 1"),  # Day too low
            ("day", 32, "less than or equal to 31"),  # Day too high
            ("duration", -1, "greater than or equal to 0"),  # Negative duration
            ("campaign", -1, "greater than or equal to 0"),  # Negative campaign
            ("previous_contact", 2, "less than or equal to 1"),  # Invalid binary
        ],
    )
    def test_prediction_input_validation_errors(self, invalid_field, invalid_value, error_match):
        """Test PredictionInput validation with invalid values."""
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
        input_data[invalid_field] = invalid_value

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            PredictionInput(**input_data)  # type: ignore[arg-type]

    def test_prediction_input_missing_required_field(self):
        """Test PredictionInput with missing required field."""
        input_data = {
            "age": 35,
            # Missing other required fields
        }

        with pytest.raises(Exception):  # Pydantic raises ValidationError
            PredictionInput(**input_data)  # type: ignore[arg-type]

    def test_prediction_output_valid_data(self):
        """Test PredictionOutput with valid data."""
        output_data = {
            "prediction": 1,
            "probability": 0.75,
            "probabilities": {"class_0": 0.25, "class_1": 0.75},
        }

        prediction_output = PredictionOutput(**output_data)  # type: ignore[arg-type]

        assert prediction_output.prediction == 1
        assert prediction_output.probability == 0.75
        assert prediction_output.probabilities == {"class_0": 0.25, "class_1": 0.75}

    def test_health_response_valid_data(self):
        """Test HealthResponse with valid data."""
        health_data = {"status": "healthy", "model_loaded": True}

        health_response = HealthResponse(**health_data)  # type: ignore[arg-type]

        assert health_response.status == "healthy"
        assert health_response.model_loaded is True

    def test_model_info_response_valid_data(self):
        """Test ModelInfoResponse with valid data."""
        info_data = {"model_name": "test_model", "model_path": "/path/to/model.pkl"}

        model_info = ModelInfoResponse(**info_data)  # type: ignore[arg-type]

        assert model_info.model_name == "test_model"
        assert model_info.model_path == "/path/to/model.pkl"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_success(self, test_config):
        """Test successful configuration loading."""
        config_path, expected_config = test_config

        with patch("src.serving.api.Path") as mock_path:
            mock_path.return_value = config_path
            config = load_config()

            assert "model" in config
            assert "api" in config
            assert config["model"]["name"] == expected_config["model"]["name"]

    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
            with pytest.raises(FileNotFoundError):
                load_config()


class TestInitializePredictor:
    """Tests for initialize_predictor function."""

    @patch("src.serving.api.load_config")
    @patch("src.serving.api.ModelPredictor")
    def test_initialize_predictor_success(self, mock_predictor_class, mock_load_config):
        """Test successful predictor initialization."""
        # Mock configuration
        mock_load_config.return_value = {
            "model": {
                "name": "test_model",
                "local_storage_path": "models/",
            }
        }

        # Mock predictor
        mock_predictor = Mock()
        mock_predictor.load_model = Mock()
        mock_predictor_class.return_value = mock_predictor

        # Initialize predictor
        predictor = initialize_predictor()

        # Verify
        assert predictor is not None
        mock_predictor.load_model.assert_called_once()

    @patch("src.serving.api.load_config")
    @patch("src.serving.api.ModelPredictor")
    def test_initialize_predictor_load_failure(self, mock_predictor_class, mock_load_config):
        """Test predictor initialization when model loading fails."""
        mock_load_config.return_value = {
            "model": {
                "name": "test_model",
                "local_storage_path": "models/",
            }
        }

        # Mock predictor that fails to load
        mock_predictor = Mock()
        mock_predictor.load_model = Mock(side_effect=Exception("Load failed"))
        mock_predictor_class.return_value = mock_predictor

        with pytest.raises(Exception, match="Load failed"):
            initialize_predictor()


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, client):
        """Test GET / root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["message"] == "Term Deposit Prediction API"
        assert isinstance(data["endpoints"], dict)
        assert "health" in data["endpoints"]
        assert "predict" in data["endpoints"]

    @patch("src.serving.api.predictor")
    def test_health_check_healthy(self, mock_predictor, client):
        """Test GET /health when model is loaded."""
        # Mock loaded predictor
        mock_predictor_instance = Mock()
        mock_predictor_instance.model = Mock()  # Model is loaded
        mock_predictor.__bool__ = lambda self: True
        mock_predictor.model = Mock()

        with patch("src.serving.api.predictor", mock_predictor_instance):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_health_check_unhealthy(self, client):
        """Test GET /health when model is not loaded."""
        with patch("src.serving.api.predictor", None):
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

    @patch("src.serving.api.predictor")
    @patch("src.serving.api.load_config")
    def test_model_info_success(self, mock_load_config, mock_predictor, client):
        """Test GET /model-info with loaded model."""
        # Mock configuration
        mock_load_config.return_value = {
            "model": {
                "name": "test_model_name",
                "local_storage_path": "models/",
            }
        }

        # Mock predictor
        mock_predictor_instance = Mock()
        mock_predictor_instance.model_path = Path("models/model.pkl")

        with patch("src.serving.api.predictor", mock_predictor_instance):
            response = client.get("/model-info")

            assert response.status_code == 200
            data = response.json()
            assert data["model_name"] == "test_model_name"
            assert "model.pkl" in data["model_path"]

    def test_model_info_model_not_loaded(self, client):
        """Test GET /model-info when model is not loaded."""
        with patch("src.serving.api.predictor", None):
            response = client.get("/model-info")

            assert response.status_code == 503
            data = response.json()
            assert "Model not loaded" in data["detail"]

    @patch("src.serving.api.predictor")
    def test_predict_success(self, mock_predictor, client):
        """Test POST /predict with valid input."""
        # Mock predictor
        mock_predictor_instance = Mock()
        mock_predictor_instance.predict = Mock(
            return_value={
                "prediction": 1,
                "probability": 0.75,
                "probabilities": {"class_0": 0.25, "class_1": 0.75},
            }
        )

        with patch("src.serving.api.predictor", mock_predictor_instance):
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

            response = client.post("/predict", json=input_data)

            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 1
            assert data["probability"] == 0.75
            assert "class_0" in data["probabilities"]
            assert "class_1" in data["probabilities"]

    def test_predict_model_not_loaded(self, client):
        """Test POST /predict when model is not loaded."""
        with patch("src.serving.api.predictor", None):
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

            response = client.post("/predict", json=input_data)

            assert response.status_code == 503
            data = response.json()
            assert "Model not loaded" in data["detail"]

    def test_predict_invalid_input(self, client):
        """Test POST /predict with invalid input data."""
        invalid_input = {
            "age": 17,  # Too young (< 18)
            "balance": 1000,
            # Missing required fields...
        }

        response = client.post("/predict", json=invalid_input)

        assert response.status_code == 422  # Unprocessable Entity (validation error)

    @patch("src.serving.api.predictor")
    def test_predict_prediction_failure(self, mock_predictor, client):
        """Test POST /predict when prediction fails."""
        # Mock predictor that raises exception
        mock_predictor_instance = Mock()
        mock_predictor_instance.predict = Mock(side_effect=Exception("Prediction failed"))

        with patch("src.serving.api.predictor", mock_predictor_instance):
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

            response = client.post("/predict", json=input_data)

            assert response.status_code == 500
            data = response.json()
            assert "Prediction failed" in data["detail"]

    @patch("src.serving.api.predictor")
    def test_predict_value_error(self, mock_predictor, client):
        """Test POST /predict when ValueError is raised."""
        # Mock predictor that raises ValueError
        mock_predictor_instance = Mock()
        mock_predictor_instance.predict = Mock(side_effect=ValueError("Invalid input format"))

        with patch("src.serving.api.predictor", mock_predictor_instance):
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

            response = client.post("/predict", json=input_data)

            assert response.status_code == 400
            data = response.json()
            assert "Invalid input" in data["detail"]


class TestAPIIntegration:
    """Integration tests for the API."""

    @patch("src.serving.api.predictor")
    def test_complete_prediction_workflow(self, mock_predictor, client):
        """Test complete workflow: health check, model info, prediction."""
        # Mock predictor
        mock_predictor_instance = Mock()
        mock_predictor_instance.model = Mock()
        mock_predictor_instance.model_path = Path("models/model.pkl")
        mock_predictor_instance.predict = Mock(
            return_value={
                "prediction": 0,
                "probability": 0.35,
                "probabilities": {"class_0": 0.65, "class_1": 0.35},
            }
        )

        with patch("src.serving.api.predictor", mock_predictor_instance):
            with patch("src.serving.api.load_config") as mock_config:
                mock_config.return_value = {
                    "model": {"name": "xgboost_model", "local_storage_path": "models/"}
                }

                # 1. Check health
                health_response = client.get("/health")
                assert health_response.status_code == 200
                assert health_response.json()["status"] == "healthy"

                # 2. Get model info
                info_response = client.get("/model-info")
                assert info_response.status_code == 200
                assert info_response.json()["model_name"] == "xgboost_model"

                # 3. Make prediction
                input_data = {
                    "age": 45,
                    "balance": 2500,
                    "day": 20,
                    "duration": 300,
                    "campaign": 1,
                    "previous": 1,
                    "total_contacts": 2,
                    "days_since_last_contact": 10,
                    "previous_contact": 1,
                    "default": "no",
                    "housing": "no",
                    "loan": "yes",
                    "job": "management",
                    "marital": "single",
                    "education": "tertiary",
                    "contact": "cellular",
                    "month": "jun",
                    "poutcome": "success",
                }

                predict_response = client.post("/predict", json=input_data)
                assert predict_response.status_code == 200
                prediction_data = predict_response.json()
                assert prediction_data["prediction"] == 0
                assert prediction_data["probability"] == 0.35
