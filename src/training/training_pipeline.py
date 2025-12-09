"""Training pipeline for machine learning model with MLflow tracking."""

import logging
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.entities.configs import TrainingDataConfig, FeatureConfig, ModelConfig
from src.utils.model_utils import (
    calculate_metrics,
    log_model_to_mlflow,
    log_class_distribution,
    log_metrics_to_mlflow,
)
from src.utils.plotting_utils import create_and_log_plots


class TrainingPipeline:
    """Pipeline for training machine learning models with preprocessing and MLflow tracking.

    This class handles the complete training workflow including:
    - Loading and validating training/test data
    - Creating sklearn preprocessing pipeline with feature transformations
    - Training the model
    - Evaluating model performance
    - Logging experiments to MLflow

    Attributes:
        job_name: Name of the training job for MLflow experiment.
        data_config: Configuration for data paths and target column.
        feature_config: Configuration for feature types and transformations.
        model_config: Configuration for model type and hyperparameters.
        pipeline: sklearn Pipeline combining preprocessing and model.
        metrics: Dictionary of evaluation metrics.
    """

    def __init__(
        self,
        job_name: str,
        data_config: TrainingDataConfig,
        feature_config: FeatureConfig,
        model_config: ModelConfig,
    ) -> None:
        """Initialize the training pipeline.

        Args:
            job_name: Name for the MLflow experiment.
            data_config: Data configuration object.
            feature_config: Feature configuration object.
            model_config: Model configuration object.
        """
        self.job_name = job_name
        self.data_config = data_config
        self.feature_config = feature_config
        self.model_config = model_config
        self.pipeline: Pipeline | None = None
        self.train_metrics: Dict[str, float] = {}
        self.test_metrics: Dict[str, float] = {}
        self.train_predictions: Dict[str, float] = {}
        self.test_predictions: Dict[str, float] = {}
        self.label_encoder: LabelEncoder | None = None

        self.logger = self._setup_logger()

        # Set up MLflow experiment
        mlflow.set_experiment(self.job_name)
        self.logger.info(f"Initialized training pipeline for experiment: {self.job_name}")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("training_pipeline")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)  # Create if doesn't exist
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

        return logger

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load training and test datasets.

        Returns:
            Tuple of (x_train, x_test, y_train, y_test) with encoded target.

        Raises:
            FileNotFoundError: If data files don't exist.
            ValueError: If target column is missing.
        """
        self.logger.info("Loading training and test data...")

        if not self.data_config.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.data_config.train_path}")
        if not self.data_config.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.data_config.test_path}")

        # Load datasets
        train_df = pd.read_csv(self.data_config.train_path)
        test_df = pd.read_csv(self.data_config.test_path)

        # Validate target column exists
        if self.data_config.target_column not in train_df.columns:
            raise ValueError(
                f"Target column '{self.data_config.target_column}' not found in training data"
            )

        # Split features and target
        x_train = train_df.drop(columns=[self.data_config.target_column])
        y_train = train_df[self.data_config.target_column]
        x_test = test_df.drop(columns=[self.data_config.target_column])
        y_test = test_df[self.data_config.target_column]

        # Encode target variable
        if self.data_config.encode_target:
            self.label_encoder = LabelEncoder()
            y_train = pd.Series(self.label_encoder.fit_transform(y_train), index=y_train.index)
            y_test = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)
            class_mapping = dict(
                zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_),
                )
            )
            self.logger.info(f"Encoded target variable: {class_mapping}")

        self.logger.info(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples")
        self.logger.info(f"Number of features: {len(x_train.columns)}")

        return x_train, x_test, y_train, y_test

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline with feature transformations.

        Returns:
            ColumnTransformer with appropriate transformations for each feature type.
        """
        self.logger.info("Creating preprocessing pipeline...")

        transformers = []

        # One-hot encoding for categorical features
        if self.feature_config.categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.feature_config.categorical_features,
                )
            )
            self.logger.info(
                f"Added OneHotEncoder for {len(self.feature_config.categorical_features)} "
                "categorical features"
            )

        # Standard scaling for numerical features
        if self.feature_config.numerical_features:
            transformers.append(
                (
                    "numerical",
                    StandardScaler(),
                    self.feature_config.numerical_features,
                )
            )
            self.logger.info(
                f"Added StandardScaler for {len(self.feature_config.numerical_features)} "
                "numerical features"
            )

        # Map binary features "yes"/"no" to 1/0
        if self.feature_config.binary_features:
            transformers.append(
                (
                    "binary",
                    OneHotEncoder(sparse_output=False, drop="if_binary"),
                    self.feature_config.binary_features,
                )
            )
            self.logger.info(
                f"Added LabelEncoder for {len(self.feature_config.binary_features)} binary features"
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        return preprocessor

    def train_model(
        self, x_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """Train the machine learning model.

        Args:
            x_train: Training features.
            y_train: Training target.

        Returns:
            Tuple of (trained pipeline, training info dictionary).
        """
        self.logger.info(f"Training {self.model_config.type} model...")

        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline()

        # Create model based on configuration
        if self.model_config.type == "XGBClassifier":
            model = XGBClassifier(**self.model_config.parameters)
        else:
            raise ValueError(f"Unsupported model type: {self.model_config.type}")

        # Create full pipeline
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # Train the model
        self.logger.info("Fitting pipeline...")
        self.pipeline.fit(x_train, y_train)

        training_info = {
            "model_type": self.model_config.type,
            "n_samples": len(x_train),
            "n_features": len(x_train.columns),
            "x_train": x_train,
        }

        self.logger.info("Model training completed successfully")
        return self.pipeline, training_info

    def evaluate_model(
        self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate model performance on train and test data.

        Args:
            x_train: Train features.
            y_train: Train target.
            x_test: Test features.
            y_test: Test target.

        Returns:
            Tuple containining (Dict of evaluation metrics on train data, Dict of eval metrics on test data).
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")

        self.logger.info("Evaluating model on test data...")

        # Make predictions on train data and calculate metrics
        y_pred_train = self.pipeline.predict(x_train)
        y_pred_proba_train = self.pipeline.predict_proba(x_train)[:, 1]
        self.train_metrics = calculate_metrics(
            y_train, y_pred_train, y_pred_proba_train, prefix="train"
        )

        # Make predictions on test data
        y_pred_test = self.pipeline.predict(x_test)
        y_pred_proba_test = self.pipeline.predict_proba(x_test)[:, 1]
        self.test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test, prefix="test")

        # Store predictions for later visualization
        self.train_predictions = {
            "y_true": y_train,
            "y_pred": y_pred_train,
            "y_pred_proba": y_pred_proba_train,
        }

        self.test_predictions = {
            "y_true": y_test,
            "y_pred": y_pred_test,
            "y_pred_proba": y_pred_proba_test,
        }

        self.logger.info("Evaluation metrics (train set):")
        for metric_name, metric_value in self.train_metrics.items():
            self.logger.info(f"  test_{metric_name}: {metric_value:.4f}")

        self.logger.info("Evaluation metrics (test set):")
        for metric_name, metric_value in self.test_metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")

        return self.train_metrics, self.test_metrics

    def log_to_mlflow(self, training_info: Dict[str, Any]) -> None:
        """Log parameters, metrics, and model to MLflow.

        Args:
            training_info: Dictionary containing training information.
        """
        self.logger.info("Logging to MLflow...")

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.model_config.parameters)
            mlflow.log_param("model_type", self.model_config.type)
            mlflow.log_param("n_train_samples", training_info["n_samples"])
            mlflow.log_param("n_features", training_info["n_features"])

            # Log feature configuration
            mlflow.log_param(
                "n_categorical_features", len(self.feature_config.categorical_features)
            )
            mlflow.log_param("n_numerical_features", len(self.feature_config.numerical_features))
            mlflow.log_param("n_binary_features", len(self.feature_config.binary_features))

            # Log train and test metrics
            log_metrics_to_mlflow(self.train_metrics, prefix="train")
            log_metrics_to_mlflow(self.test_metrics, prefix="test")

            # Log plots
            create_and_log_plots(
                self.test_predictions["y_true"],
                self.test_predictions["y_pred"],
                self.test_predictions["y_pred_proba"],
                self.pipeline,
            )

            # Log class distribution
            log_class_distribution(
                self.train_predictions.get("y_true", []), self.test_predictions.get("y_true", [])
            )

            # Log model (includes preprocessing pipeline)
            if self.pipeline is not None:
                log_model_to_mlflow(self.pipeline, model_name=self.model_config.type)

            # Log config file
            mlflow.log_artifact("confs/training.yaml", artifact_path="config")

            self.logger.info("Successfully logged to MLflow")

    def run(self) -> Tuple[Pipeline, Dict[str, float]]:
        """Execute the complete training pipeline.

        Returns:
            Tuple of (trained pipeline, evaluation metrics).
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training pipeline: {self.job_name}")
        self.logger.info("=" * 80)

        try:
            # Load data
            x_train, x_test, y_train, y_test = self.load_data()

            # Train model
            pipeline, training_info = self.train_model(x_train, y_train)

            # Evaluate model on test data
            _, test_metrics = self.evaluate_model(x_train, y_train, x_test, y_test)

            # Log to MLflow
            self.log_to_mlflow(training_info)

            self.logger.info("=" * 80)
            self.logger.info("Training pipeline completed successfully!")
            self.logger.info("=" * 80)

            return pipeline, test_metrics

        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise
