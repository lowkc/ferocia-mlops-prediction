"""Shared modelling utilities for model evaluation and logging."""

from typing import Dict, Optional
import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray, prefix: Optional[str] = None
) -> Dict[str, float]:
    """Calculate standard classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class
        prefix: Optional prefix for metric names (e.g., "train_", "test_")

    Returns:
        Dictionary of metrics
    """
    base_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in base_metrics.items()}
    return base_metrics


def log_model_to_mlflow(
    pipeline: Pipeline,
    model_name: str,
    x_train: Optional[pd.DataFrame] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    """Log sklearn pipeline to MLflow with signature.

    Args:
        pipeline: Trained sklearn pipeline
        model_name: Artifact path name
        x_train: Training data for signature inference
        registered_model_name: Name for model registry
    """
    signature = None
    if x_train is not None and not x_train.empty:
        signature = infer_signature(x_train, pipeline.predict(x_train))

    mlflow.sklearn.log_model(
        pipeline,
        name=model_name,
        registered_model_name=registered_model_name,
        signature=signature,
    )


def log_metrics_to_mlflow(metrics: Dict[str, float], prefix: Optional[str] = None) -> None:
    """Log metrics dictionary to MLflow.

    Args:
        metrics: Dictionary of metric_name: metric_value
        prefix: Optional prefix for metric names (e.g., "train_", "test_")
    """
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    mlflow.log_metrics(metrics)


def log_class_distribution(
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Log class distribution to MLflow parameters.

    Args:
        y_train: Training labels
        y_test: Test labels
    """
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    total_train = len(y_train)
    total_test = len(y_test)

    mlflow.log_param("train_class_0_count", train_counts.get(0, 0))
    mlflow.log_param("train_class_1_count", train_counts.get(1, 0))
    mlflow.log_param("train_class_0_pct", train_counts.get(0, 0) / total_train * 100)
    mlflow.log_param("train_class_1_pct", train_counts.get(1, 0) / total_train * 100)

    mlflow.log_param("test_class_0_count", test_counts.get(0, 0))
    mlflow.log_param("test_class_1_count", test_counts.get(1, 0))
    mlflow.log_param("test_class_0_pct", test_counts.get(0, 0) / total_test * 100)
    mlflow.log_param("test_class_1_pct", test_counts.get(1, 0) / total_test * 100)
