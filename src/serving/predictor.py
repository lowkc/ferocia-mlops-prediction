"""Model predictor class for loading and making predictions."""

import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


class ModelPredictor:
    """Class for loading ML model and making predictions.

    This class handles model loading and prediction with preprocessing.
    The model is expected to be a scikit-learn Pipeline that includes
    preprocessing steps (OneHotEncoder, StandardScaler, etc.) and the
    classifier.

    Attributes:
        model_path: Path to the serialized model file.
        model: Loaded sklearn Pipeline model.
        logger: Logger instance for tracking operations.
    """

    def __init__(self, model_path: Path) -> None:
        """Initialize ModelPredictor.

        Args:
            model_path: Path to the model pickle file.
        """
        self.model_path = model_path
        self.model = None
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger("model_predictor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def load_model(self) -> None:
        """Load the trained model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            Exception: If model loading fails.
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info("Model loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions on input data.

        This method accepts raw input data, applies any necessary preprocessing
        (which is included in the sklearn Pipeline), and returns predictions.

        Args:
            input_data: Dictionary containing input features. Should match the
                       schema expected by the model's preprocessing pipeline.

        Returns:
            Dictionary containing:
                - prediction: Binary prediction (0 or 1)
                - probability: Probability of positive class
                - probabilities: Array of probabilities for all classes

        Raises:
            ValueError: If model hasn't been loaded or input is invalid.
            Exception: If prediction fails.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")

        try:
            self.logger.info("Making prediction on input data")

            # Convert input to DataFrame
            # The model expects a DataFrame with specific columns
            df = pd.DataFrame([input_data])

            self.logger.debug(f"Input DataFrame shape: {df.shape}")
            self.logger.debug(f"Input columns: {df.columns.tolist()}")

            # Make prediction
            # The model is a Pipeline that handles preprocessing internally
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]

            # Prepare result
            result = {
                "prediction": int(prediction),
                "probability": float(probabilities[1]),  # Probability of positive class
                "probabilities": {
                    "class_0": float(probabilities[0]),
                    "class_1": float(probabilities[1]),
                },
            }

            self.logger.info(f"Prediction: {prediction}, Probability: {probabilities[1]:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
