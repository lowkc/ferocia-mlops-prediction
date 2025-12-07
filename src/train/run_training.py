"""Command-line script to run the model training pipeline."""

import argparse
import sys
from pathlib import Path

from train.config import load_config
from train.training_pipeline import TrainingPipeline


def main() -> int:
    """Main entry point for training script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Train ML model with MLflow tracking")
    parser.add_argument(
        "--config",
        type=str,
        default="confs/train.yaml",
        help="Path to configuration YAML file (default: confs/train.yaml)",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        job_name, data_config, feature_config, model_config = load_config(args.config)
        print(f"\nLoaded configuration from: {args.config}")
        print(f"Experiment name: {job_name}")

    except Exception as e:
        print(f"\nError loading configuration: {e}", file=sys.stderr)
        return 1

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Run training pipeline
    try:
        pipeline = TrainingPipeline(
            job_name=job_name,
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
        )

        trained_pipeline, metrics = pipeline.run()

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nModel: {model_config.type}")
        print(f"\nTraining data: {data_config.train_path}")
        print(f"Test data: {data_config.test_path}")
        print("\nEvaluation Metrics:")
        print("-" * 40)
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name:.<30} {metric_value:.4f}")
        print("\nMLflow:")
        print(f"  Experiment: {job_name}")
        print("  Run logged successfully")
        print("\nLogs saved to: logs/training.log")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nTraining failed. Check logs/training.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
