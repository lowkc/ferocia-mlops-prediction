"""Plotting utilities for model evaluation during training and hyperparmeter tuning."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.pipeline import Pipeline


def save_plot(output_path: str, dpi: int = 150) -> str:
    """Save the current matplotlib figure to a file.

    Args:
        output_path: Path where the figure should be saved (e.g., "outputs/plot.png").
        dpi: Resolution in dots per inch. Default is 150.

    Returns:
        Path to the saved figure.
    """
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save and close the figure
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, title: str = "Confusion Matrix"
) -> str:
    """Create and save confusion matrix plot.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Title for the plot.

    Returns:
        Path to saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            plt.text(
                j + 0.5,
                i + 0.7,
                f"({cm[i, j] / total * 100:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    fig_path = "outputs/confusion_matrix.png"
    return save_plot(fig_path)


def plot_roc_curve(y_true: pd.Series, y_pred_proba: np.ndarray, title: str = "ROC Curve") -> str:
    """Create and save ROC curve plot.

    Args:
        y_true: True labels.
        y_pred_proba: Predicted probabilities for positive class.
        title: Title for the plot.

    Returns:
        Path to saved figure.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    fig_path = "outputs/roc_curve.png"
    return save_plot(fig_path)


def plot_feature_importance(pipeline: Pipeline, top_n: int = 20) -> str:
    """Create and save feature importance plot for XGBoost model.

    Args:
        pipeline: Trained sklearn pipeline with XGBoost classifier.
        top_n: Number of top features to display.

    Returns:
        Path to saved figure.
    """
    # Extract the classifier from pipeline
    model = pipeline.named_steps["classifier"]

    # Get feature importance
    importance = model.feature_importances_

    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "categorical":
            # OneHotEncoder creates multiple features
            if hasattr(transformer, "get_feature_names_out"):
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                feature_names.extend(columns)
        else:
            feature_names.extend(columns)

    # Create DataFrame and sort
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df["importance"].values)
    plt.yticks(range(len(importance_df)), importance_df["feature"].values)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    fig_path = "outputs/feature_importance.png"
    return save_plot(fig_path)
