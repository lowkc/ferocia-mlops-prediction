"""Script to download MLflow model from tracking server to local storage."""

import argparse
import logging
from pathlib import Path

import mlflow
import yaml


def setup_logger() -> logging.Logger:
    """Setup logging configuration.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("download_mlflow_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_deployment_config(config_path: Path) -> dict:
    """Load deployment configuration from YAML file.

    Args:
        config_path: Path to deployment config YAML file.

    Returns:
        Dictionary containing deployment configuration.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def download_model(model_name: str, local_storage_path: Path, logger: logging.Logger) -> None:
    """Download model from MLflow model registry to local storage.

    Args:
        model_name: Name of the registered model in MLflow.
        local_storage_path: Local directory to save the model.
        logger: Logger instance.

    Raises:
        Exception: If model download fails.
    """
    try:
        logger.info(f"Downloading model '{model_name}' from MLflow...")

        # Create local storage directory if it doesn't exist
        local_storage_path.mkdir(parents=True, exist_ok=True)

        # Get the latest version of the model
        client = mlflow.MlflowClient()

        try:
            # Try to get from model registry first
            model_versions = client.search_model_versions(f"name='{model_name}'")

            if not model_versions:
                logger.warning(
                    f"Model '{model_name}' not found in registry. Searching in experiments..."
                )

                # Search for the model in experiments by name
                experiments = client.search_experiments()
                model_uri = None

                for exp in experiments:
                    runs = client.search_runs(
                        experiment_ids=[exp.experiment_id],
                        order_by=["metrics.test_f1 DESC"],
                        max_results=1,
                    )

                    if runs:
                        run = runs[0]
                        # Check if this run has the model we're looking for
                        artifacts = client.list_artifacts(run.info.run_id)
                        for artifact in artifacts:
                            if artifact.path == "XGBClassifier":
                                model_uri = f"runs:/{run.info.run_id}/XGBClassifier"
                                logger.info(f"Found model in run: {run.info.run_id}")
                                break

                    if model_uri:
                        break

                if not model_uri:
                    raise ValueError(f"Could not find model '{model_name}' in MLflow")

            else:
                # Get the latest version
                latest_version = model_versions[0]
                logger.info(
                    f"Found model version: {latest_version.version} "
                    f"(stage: {latest_version.current_stage})"
                )
                model_uri = f"models:/{model_name}/{latest_version.version}"

        except Exception as e:
            logger.warning(f"Error accessing model registry: {e}")
            logger.info("Attempting to find model in recent runs...")

            # Fallback: Search for the best model in the most recent runs
            experiments = client.search_experiments()
            model_uri = None
            best_f1 = 0

            for exp in experiments:
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="metrics.test_f1 > 0",
                    order_by=["metrics.test_f1 DESC"],
                    max_results=5,
                )

                for run in runs:
                    f1_score = run.data.metrics.get("test_f1", 0)
                    if f1_score > best_f1:
                        # Check if this run has a model artifact
                        artifacts = client.list_artifacts(run.info.run_id)
                        for artifact in artifacts:
                            if artifact.path == "XGBClassifier":
                                model_uri = f"runs:/{run.info.run_id}/XGBClassifier"
                                best_f1 = f1_score
                                logger.info(
                                    f"Found model in run {run.info.run_id} with F1: {f1_score:.4f}"
                                )
                                break

            if not model_uri:
                raise ValueError("Could not find any trained model in MLflow")

        # Download the model
        model_path = local_storage_path / "model"
        logger.info(f"Downloading model from: {model_uri}")
        logger.info(f"Saving to: {model_path}")

        # Save the sklearn model for easier loading
        model = mlflow.sklearn.load_model(model_uri)
        import joblib

        joblib.dump(model, local_storage_path / "model.pkl")

        logger.info(f"Successfully downloaded model to {local_storage_path}")
        logger.info(f"Model pickle saved to: {local_storage_path / 'model.pkl'}")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """Main function to download MLflow model."""
    parser = argparse.ArgumentParser(description="Download MLflow model to local storage")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("confs/deployment.yaml"),
        help="Path to deployment configuration file",
    )
    parser.add_argument("--version", type=str, default="latest", help="Model version")

    args = parser.parse_args()
    logger = setup_logger()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_deployment_config(args.config)

        model_config = config.get("model", {})
        model_name = model_config.get("name")
        version = args.version
        local_storage_path = Path(model_config.get("local_storage_path", "models/"))

        if not model_name:
            raise ValueError("Model name not specified in configuration")

        logger.info("Configuration loaded:")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Model version: {version}")
        logger.info(f"  Local storage path: {local_storage_path}")

        # Download model
        download_model(model_name, local_storage_path, logger)

        logger.info("=" * 80)
        logger.info("Model download completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise


if __name__ == "__main__":
    main()
