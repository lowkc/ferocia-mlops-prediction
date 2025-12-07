"""Training pipeline for machine learning model with MLflow tracking."""

import logging
from typing import Any, Dict, Tuple

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from train.config import DataConfig, FeatureConfig, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


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
        data_config: DataConfig,
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
        self.metrics: Dict[str, float] = {}
        self.label_encoder: LabelEncoder | None = None

        # Set up MLflow experiment
        mlflow.set_experiment(self.job_name)
        logger.info(f"Initialized training pipeline for experiment: {self.job_name}")

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load training and test datasets.

        Returns:
            Tuple of (x_train, x_test, y_train, y_test) with encoded target.

        Raises:
            FileNotFoundError: If data files don't exist.
            ValueError: If target column is missing.
        """
        logger.info("Loading training and test data...")

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
            logger.info(f"Encoded target variable: {class_mapping}")

        logger.info(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples")
        logger.info(f"Number of features: {len(x_train.columns)}")

        return x_train, x_test, y_train, y_test

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create sklearn preprocessing pipeline with feature transformations.

        Returns:
            ColumnTransformer with appropriate transformations for each feature type.
        """
        logger.info("Creating preprocessing pipeline...")

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
            logger.info(
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
            logger.info(
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
            logger.info(
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
        logger.info(f"Training {self.model_config.type} model...")

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
        logger.info("Fitting pipeline...")
        self.pipeline.fit(x_train, y_train)

        training_info = {
            "model_type": self.model_config.type,
            "n_samples": len(x_train),
            "n_features": len(x_train.columns),
            "x_train": x_train,
        }

        logger.info("Model training completed successfully")
        return self.pipeline, training_info

    def evaluate_model(self, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on test data.

        Args:
            x_test: Test features.
            y_test: Test target.

        Returns:
            Dictionary of evaluation metrics.
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model on test data...")

        # Make predictions
        y_pred = self.pipeline.predict(x_test)
        y_pred_proba = self.pipeline.predict_proba(x_test)[:, 1]

        # Calculate metrics
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.info("Evaluation metrics:")
        for metric_name, metric_value in self.metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

        return self.metrics

    def log_to_mlflow(self, training_info: Dict[str, Any]) -> None:
        """Log parameters, metrics, and model to MLflow.

        Args:
            training_info: Dictionary containing training information.
        """
        logger.info("Logging to MLflow...")

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

            # Log metrics
            mlflow.log_metrics(self.metrics)

            # Infer signature
            signature = None
            x_train = training_info.get("x_train", pd.DataFrame())
            if not x_train.empty and self.pipeline:
                signature = infer_signature(x_train, self.pipeline.predict(x_train))

            # Log model (includes preprocessing pipeline)
            if self.pipeline is not None:
                mlflow.sklearn.log_model(
                    self.pipeline,
                    name=self.model_config.type,
                    registered_model_name=self.job_name,
                    signature=signature,
                )

            # Log config file
            mlflow.log_artifact("confs/train.yaml", artifact_path="config")

            logger.info("Successfully logged to MLflow")

    def run(self) -> Tuple[Pipeline, Dict[str, float]]:
        """Execute the complete training pipeline.

        Returns:
            Tuple of (trained pipeline, evaluation metrics).
        """
        logger.info("=" * 80)
        logger.info(f"Starting training pipeline: {self.job_name}")
        logger.info("=" * 80)

        try:
            # Load data
            x_train, x_test, y_train, y_test = self.load_data()

            # Train model
            pipeline, training_info = self.train_model(x_train, y_train)

            # Evaluate model
            metrics = self.evaluate_model(x_test, y_test)

            # Log to MLflow
            self.log_to_mlflow(training_info)

            logger.info("=" * 80)
            logger.info("Training pipeline completed successfully!")
            logger.info("=" * 80)

            return pipeline, metrics

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
