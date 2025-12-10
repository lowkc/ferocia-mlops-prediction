"""FastAPI application for model serving."""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.serving.predictor import ModelPredictor


# Pydantic models for request/response validation
class PredictionInput(BaseModel):
    """Input schema for prediction requests.

    This schema matches the expected features from the bank marketing dataset
    after preprocessing. All features are required for prediction.
    """

    # Numerical features
    age: int = Field(..., description="Age of the client", ge=18, le=100)
    balance: int = Field(..., description="Average yearly balance")
    day: int = Field(..., description="Last contact day of the month", ge=1, le=31)
    duration: int = Field(..., description="Last contact duration in seconds", ge=0)
    campaign: int = Field(
        ..., description="Number of contacts performed during this campaign", ge=0
    )
    previous: int = Field(
        ..., description="Number of contacts performed before this campaign", ge=0
    )
    total_contacts: int = Field(
        ..., description="Total number of contacts (campaign + previous)", ge=0
    )
    days_since_last_contact: int = Field(
        ..., description="Number of days since client was last contacted (0 if not contacted)", ge=0
    )

    # Binary features
    previous_contact: int = Field(..., description="Previous contact flag (0 or 1)", ge=0, le=1)
    default: str = Field(..., description="Has credit in default?")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")

    # Categorical features
    job: str = Field(..., description="Type of job")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    contact: str = Field(..., description="Contact communication type")
    month: str = Field(..., description="Last contact month of year")
    poutcome: str = Field(..., description="Outcome of the previous marketing campaign")

    class Config:
        """Pydantic config."""

        populate_by_name = True
        json_schema_extra = {
            "example": {
                "age": 56,
                "balance": 1234,
                "day": 5,
                "duration": 261,
                "campaign": 1,
                "previous": 0,
                "total_contacts": 1,
                "days_since_last_contact": 0,
                "previous_contact": 0,
                "default": "no",
                "housing": "no",
                "loan": "no",
                "job": "housemaid",
                "marital": "married",
                "education": "basic.4y",
                "contact": "telephone",
                "month": "may",
                "poutcome": "nonexistent",
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""

    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(
        ..., description="Probability of positive class (term deposit subscription)"
    )
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_name: str = Field(..., description="Name of the loaded model")
    model_path: str = Field(..., description="Path to the model file")


# Initialize FastAPI app
app = FastAPI(
    title="Term Deposit Prediction API",
    description="API for predicting term deposit subscriptions using XGBoost",
    version="1.0.0",
)

# Global predictor instance
predictor: ModelPredictor | None = None


def load_config() -> Dict[str, Any]:
    """Load deployment configuration.

    Returns:
        Dictionary containing deployment configuration.
    """
    config_path = Path("confs/deployment.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def initialize_predictor() -> ModelPredictor:
    """Initialize and load the model predictor.

    Returns:
        Initialized ModelPredictor instance.

    Raises:
        Exception: If model initialization fails.
    """
    try:
        config = load_config()
        model_config = config.get("model", {})
        model_path = Path(model_config.get("local_storage_path", "models/")) / "model.pkl"

        pred = ModelPredictor(model_path)
        pred.load_model()
        return pred

    except Exception as e:
        logging.error(f"Failed to initialize predictor: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize model on application startup."""
    global predictor
    try:
        predictor = initialize_predictor()
        logging.info("Model loaded successfully on startup")
    except Exception as e:
        logging.error(f"Failed to load model on startup: {e}")
        # Don't fail startup, but predictor will be None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Term Deposit Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns:
        Service health status and model loading status.
    """
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None and predictor.model is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get information about the loaded model.

    Returns:
        Model metadata.

    Raises:
        HTTPException: If model is not loaded.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    config = load_config()
    model_config = config.get("model", {})

    return ModelInfoResponse(
        model_name=model_config.get("name", "unknown"),
        model_path=str(predictor.model_path),
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make a prediction on input data.

    This endpoint accepts raw input features and returns a prediction.
    The preprocessing is handled automatically by the model pipeline.

    Args:
        input_data: Input features matching the PredictionInput schema.

    Returns:
        Prediction result with probability scores.

    Raises:
        HTTPException: If prediction fails or model is not loaded.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert Pydantic model to dict, handling aliases
        input_dict = input_data.model_dump(by_alias=True)

        # Make prediction
        result = predictor.predict(input_dict)

        return PredictionOutput(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Load configuration
    config = load_config()
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)

    # Run the API
    uvicorn.run(app, host=host, port=port)
