"""Model hyperparameter tuning module using Optuna."""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import mlflow
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.entities.configs import (
    FeatureConfig,
    ModelConfig,
    TrainingDataConfig,
    TuningConfig,
)
from src.training.training_pipeline import TrainingPipeline
from src.utils.model_utils import calculate_metrics, log_model_to_mlflow, log_class_distribution
from src.utils.plotting_utils import create_and_log_plots


class HyperparameterTuningPipeline:
    """Pipeline for hyperparameter tuning using Optuna with MLflow tracking.

    This class handles the complete hyperparameter tuning workflow including:
    - Loading training data and creating preprocessing pipeline
    - Setting up Optuna study with configurable search spaces
    - Performing 5-fold cross-validation for each trial
    - Logging experiments to MLflow with parent-child run hierarchy
    - Tracking both train and validation metrics for each fold

    Attributes:
        job_name: Name of the tuning job for MLflow experiment.
        data_config: Configuration for data paths and target column.
        feature_config: Configuration for feature types and transformations.
        model_config: Configuration for base model type and fixed parameters.
        tuning_config: Configuration for hyperparameter search spaces and study settings.
        logger: Logger instance for tracking pipeline execution.
    """

    def __init__(
        self,
        job_name: str,
        data_config: TrainingDataConfig,
        feature_config: FeatureConfig,
        model_config: ModelConfig,
        tuning_config: TuningConfig,
    ) -> None:
        """Initialize the hyperparameter tuning pipeline.

        Args:
            job_name: Name for the MLflow experiment.
            data_config: Data configuration object.
            feature_config: Feature configuration object.
            model_config: Model configuration object with base parameters.
            tuning_config: Tuning configuration object with search spaces.
        """
        self.job_name = f"{job_name}_tuning"
        self.data_config = data_config
        self.feature_config = feature_config
        self.model_config = model_config
        self.tuning_config = tuning_config

        self.logger = self._setup_logger()

        # Create a training pipeline instance to reuse methods
        self.training_pipeline = TrainingPipeline(
            job_name=job_name,
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        # Set up MLflow experiment
        mlflow.set_experiment(self.job_name)
        self.logger.info(f"Initialized tuning pipeline for experiment: {self.job_name}")

        # Store data for use in objective function
        self.x_train: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.x_test: pd.DataFrame | None = None
        self.y_test: pd.Series | None = None

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("hyperparameter_tuning")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "tuning.log")
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

        return logger

    def load_data(self) -> None:
        """Load training and test datasets using the training pipeline."""
        self.logger.info("Loading training and test data...")
        x_train, x_test, y_train, y_test = self.training_pipeline.load_data()

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.logger.info(f"Loaded {len(x_train)} training samples")

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the current trial based on tuning config.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        suggested_params = {}

        for param_name, param_config in self.tuning_config.params.items():
            if not isinstance(param_config, dict):
                # Fixed parameter (not a search space)
                suggested_params[param_name] = param_config
                continue

            param_type = param_config.get("type")

            if param_type == "categorical":
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_config["choices"]
                )
            elif param_type == "int":
                suggested_params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1),
                )
            elif param_type == "float":
                # Ensure low and high are float, not strings
                low = float(param_config["low"])
                high = float(param_config["high"])
                if param_config.get("log", False):
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        low,
                        high,
                        log=True,
                    )
                else:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        low,
                        high,
                    )
            else:
                self.logger.warning(f"Unknown parameter type '{param_type}' for {param_name}")

        return suggested_params

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function with 5-fold cross-validation.

        This function:
        1. Suggests hyperparameters for the current trial
        2. Performs 5-fold stratified cross-validation
        3. Logs metrics for each fold to MLflow
        4. Returns the mean validation metric for optimization

        Args:
            trial: Optuna trial object.

        Returns:
            Mean validation metric across all folds.
        """
        # Suggest hyperparameters
        suggested_params = self._suggest_hyperparameters(trial)

        # Merge with fixed parameters from model config
        all_params = {**self.model_config.parameters, **suggested_params}

        self.logger.info(f"Trial {trial.number}: Testing parameters {suggested_params}")

        # Start nested MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Log trial parameters
            mlflow.log_params(suggested_params)
            mlflow.log_param("trial_number", trial.number)

            # 5-fold stratified cross-validation
            cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.tuning_config.random_state
            )

            fold_train_metrics: Dict[str, list] = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "roc_auc": [],
            }
            fold_val_metrics: Dict[str, list] = {
                "accuracy": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "roc_auc": [],
            }

            for fold_idx, (train_idx, val_idx) in enumerate(
                cv.split(self.x_train, self.y_train), 1
            ):
                assert (self.x_train is not None) and (self.y_train is not None), (
                    "Training data (x_train and y_train) must be initialized before splitting folds."
                )
                # Split data for this fold
                x_train_fold = self.x_train.iloc[train_idx]
                y_train_fold = self.y_train.iloc[train_idx]
                x_val_fold = self.x_train.iloc[val_idx]
                y_val_fold = self.y_train.iloc[val_idx]

                # Create preprocessing pipeline
                preprocessor = self.training_pipeline.create_preprocessing_pipeline()

                # Create model with suggested parameters
                if self.model_config.type == "XGBClassifier":
                    model = XGBClassifier(**all_params)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_config.type}")

                # Create and train pipeline
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", model),
                    ]
                )

                pipeline.fit(x_train_fold, y_train_fold)

                # Calculate train metrics
                y_train_pred = pipeline.predict(x_train_fold)
                y_train_pred_proba = pipeline.predict_proba(x_train_fold)[:, 1]
                train_metrics = calculate_metrics(y_train_fold, y_train_pred, y_train_pred_proba)

                # Calculate validation metrics
                y_val_pred = pipeline.predict(x_val_fold)
                y_val_pred_proba = pipeline.predict_proba(x_val_fold)[:, 1]
                val_metrics = calculate_metrics(y_val_fold, y_val_pred, y_val_pred_proba)

                # Store metrics for aggregation
                for metric_name in fold_train_metrics.keys():
                    fold_train_metrics[metric_name].append(train_metrics[metric_name])
                    fold_val_metrics[metric_name].append(val_metrics[metric_name])

                # Log individual fold metrics
                mlflow.log_metrics(
                    {f"fold_{fold_idx}_train_{k}": v for k, v in train_metrics.items()}
                )
                mlflow.log_metrics({f"fold_{fold_idx}_val_{k}": v for k, v in val_metrics.items()})

                self.logger.info(
                    f"  Fold {fold_idx}/5 - Val F1: {val_metrics['f1_score']:.4f}, "
                    f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
                )

            # Calculate and log aggregated metrics (mean ± std)
            aggregated_metrics = {}
            for metric_name in fold_train_metrics.keys():
                train_mean = np.mean(fold_train_metrics[metric_name])
                train_std = np.std(fold_train_metrics[metric_name])
                val_mean = np.mean(fold_val_metrics[metric_name])
                val_std = np.std(fold_val_metrics[metric_name])

                aggregated_metrics[f"train_{metric_name}_mean"] = train_mean
                aggregated_metrics[f"train_{metric_name}_std"] = train_std
                aggregated_metrics[f"val_{metric_name}_mean"] = val_mean
                aggregated_metrics[f"val_{metric_name}_std"] = val_std

            mlflow.log_metrics(aggregated_metrics)

            # Log the optimization metric
            optimization_metric = aggregated_metrics["val_f1_score_mean"]
            self.logger.info(
                f"Trial {trial.number} completed - Mean Val F1: {optimization_metric:.4f}"
            )

            return optimization_metric

    def run(self) -> Dict[str, Any]:
        """Execute the complete hyperparameter tuning pipeline.

        Returns:
            Dictionary containing best parameters and metrics.
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting hyperparameter tuning: {self.job_name}")
        self.logger.info("=" * 80)

        try:
            # Load data
            self.load_data()

            # Create parent MLflow run for the entire study
            with mlflow.start_run(run_name="hyperparameter_tuning_study"):
                # Log study configuration
                mlflow.log_param("n_trials", self.tuning_config.n_trials)
                mlflow.log_param("direction", self.tuning_config.direction)
                mlflow.log_param("random_state", self.tuning_config.random_state)
                mlflow.log_param("cv_folds", 5)
                mlflow.log_param("model_type", self.model_config.type)

                # Log fixed model parameters
                for param_name, param_value in self.model_config.parameters.items():
                    mlflow.log_param(f"fixed_{param_name}", param_value)

                # Create and run Optuna study
                self.logger.info(f"Creating Optuna study with {self.tuning_config.n_trials} trials")
                study = optuna.create_study(
                    direction=self.tuning_config.direction,
                    sampler=optuna.samplers.TPESampler(seed=self.tuning_config.random_state),
                )

                study.optimize(
                    self._objective,
                    n_trials=self.tuning_config.n_trials,
                    show_progress_bar=True,
                )

                # Log best results
                best_params = study.best_params
                best_value = study.best_value

                self.logger.info("=" * 80)
                self.logger.info("HYPERPARAMETER TUNING COMPLETED!")
                self.logger.info("=" * 80)
                self.logger.info(f"Best trial: {study.best_trial.number}")
                self.logger.info(f"Best validation F1 score: {best_value:.4f}")
                self.logger.info("Best hyperparameters:")
                for param_name, param_value in best_params.items():
                    self.logger.info(f"  {param_name}: {param_value}")

                # Log best parameters and value to MLflow
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("best_val_f1_score", best_value)

                # Evaluate best model on test set
                self.logger.info("\nEvaluating best model on test set...")
                best_all_params = {**self.model_config.parameters, **best_params}

                # Create and train final model with best parameters
                preprocessor = self.training_pipeline.create_preprocessing_pipeline()
                if self.model_config.type == "XGBClassifier":
                    model = XGBClassifier(**best_all_params)
                else:
                    raise ValueError(f"Unsupported model type: {self.model_config.type}")

                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", model),
                    ]
                )

                pipeline.fit(self.x_train, self.y_train)

                # Evaluate on test set
                y_test_pred = pipeline.predict(self.x_test)
                y_test_pred_proba = pipeline.predict_proba(self.x_test)[:, 1]
                test_metrics = calculate_metrics(self.y_test, y_test_pred, y_test_pred_proba)

                # Log test metrics
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

                self.logger.info("\nTest Set Metrics:")
                self.logger.info("-" * 40)
                for metric_name, metric_value in test_metrics.items():
                    self.logger.info(f"  {metric_name:.<30} {metric_value:.4f}")

                # Create and log visualizations
                self.logger.info("\nCreating and logging visualizations...")
                create_and_log_plots(self.y_test, y_test_pred, y_test_pred_proba, pipeline)

                # Create plots for Optuna optimization history
                try:
                    opt_hist_path = self._plot_optimization_history(study)
                    mlflow.log_artifact(opt_hist_path, artifact_path="optuna")
                    self.logger.info(f"Optimization history saved to {opt_hist_path}")
                except Exception as e:
                    self.logger.warning(f"Could not create optimization history: {e}")

                # Parameter Importances
                param_imp_path = self._plot_param_importances(study)
                if param_imp_path:
                    mlflow.log_artifact(param_imp_path, artifact_path="optuna")
                    self.logger.info(f"Parameter importances saved to {param_imp_path}")

                # Parameter Slice Plot
                param_slice_path = self._plot_param_slice(study)
                if param_slice_path:
                    mlflow.log_artifact(param_slice_path, artifact_path="optuna")
                    self.logger.info(f"Parameter slice plot saved to {param_slice_path}")

                # Log the best model
                try:
                    log_model_to_mlflow(pipeline, model_name=f"{self.job_name}_best_model")
                    self.logger.info("Best model logged to MLflow")
                except Exception as e:
                    self.logger.warning(f"Could not log model: {e}")

                # Log class distribution information
                log_class_distribution(self.y_train, self.y_test)

                # Log config files
                mlflow.log_artifact("confs/training.yaml", artifact_path="config")
                mlflow.log_artifact("confs/tuning.yaml", artifact_path="config")

                # Log study statistics
                mlflow.log_param("n_completed_trials", len(study.trials))
                mlflow.log_param(
                    "n_pruned_trials",
                    len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                )
                mlflow.log_param(
                    "n_failed_trials",
                    len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                )

                self.logger.info("\nMLflow:")
                self.logger.info(f"  Experiment: {self.job_name}")
                self.logger.info("  Study logged successfully")
                self.logger.info("  Visualizations and artifacts logged")
                self.logger.info("\nLogs saved to: logs/tuning.log")
                self.logger.info("=" * 80)

                return {
                    "best_params": best_params,
                    "best_value": best_value,
                    "test_metrics": test_metrics,
                    "study": study,
                }

        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            raise

    #### Hyperparameter tuning specific plotting tools
    def _plot_optimization_history(self, study: optuna.Study) -> str:
        """Create and save Optuna optimization history plot.

        Args:
            study: Completed Optuna study.

        Returns:
            Path to saved figure.
        """
        fig = plot_optimization_history(study)
        fig_path = "outputs/optimization_history.html"
        fig.write_html(fig_path)
        return fig_path

    def _plot_param_importances(self, study: optuna.Study) -> str | None:
        """Create and save parameter importance plot.

        Args:
            study: Completed Optuna study.

        Returns:
            Path to saved figure.
        """
        try:
            fig = plot_param_importances(study)
            fig_path = "outputs/param_importances.html"
            fig.write_html(fig_path)
            return fig_path
        except Exception as e:
            self.logger.warning(f"Could not create parameter importance plot: {e}")
            return None

    def _plot_param_slice(self, study: optuna.Study) -> str | None:
        """Create and save parameter slice plot.

        Args:
            study: Completed Optuna study.

        Returns:
            Path to saved figure.
        """
        try:
            fig = plot_slice(study)
            fig_path = "outputs/param_slice.html"
            fig.write_html(fig_path)
            return fig_path
        except Exception as e:
            self.logger.warning(f"Could not create parameter slice plot: {e}")
            return None
