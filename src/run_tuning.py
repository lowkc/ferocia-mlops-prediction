"""Command-line script to run the hyperparameter tuning pipeline."""

import argparse
import sys
from pathlib import Path

from src.training.config import load_tuning_config
from src.training.hyperparameter_tuning import HyperparameterTuningPipeline


def main() -> int:
    """Main entry point for hyperparameter tuning script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter tuning with Optuna and MLflow tracking"
    )
    parser.add_argument(
        "--tuning-config",
        type=str,
        default="confs/tuning.yaml",
        help="Path to tuning configuration YAML file (default: confs/tuning.yaml)",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="confs/training.yaml",
        help="Path to training configuration YAML file (default: confs/training.yaml)",
    )

    args = parser.parse_args()

    # Load configuration from YAML files
    try:
        (
            job_name,
            data_config,
            feature_config,
            model_config,
            tuning_config,
        ) = load_tuning_config(args.tuning_config, args.training_config)
        print(f"\nLoaded tuning configuration from: {args.tuning_config}")
        print(f"Loaded training configuration from: {args.training_config}")
        print(f"Experiment name: {job_name}_tuning")

    except Exception as e:
        print(f"\nError loading configuration: {e}", file=sys.stderr)
        return 1

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Run hyperparameter tuning pipeline
    try:
        pipeline = HyperparameterTuningPipeline(
            job_name=job_name,
            data_config=data_config,
            feature_config=feature_config,
            model_config=model_config,
            tuning_config=tuning_config,
        )

        results = pipeline.run()

        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nModel: {model_config.type}")
        print(f"Optimization trials: {tuning_config.n_trials}")
        print("Cross-validation folds: 5")
        print(f"\nTraining data: {data_config.train_path}")
        print(f"Test data: {data_config.test_path}")
        print("\nBest Hyperparameters:")
        print("-" * 40)
        for param_name, param_value in results["best_params"].items():
            print(f"  {param_name:.<30} {param_value}")
        print(f"\nBest Validation F1 Score: {results['best_value']:.4f}")
        print("\nTest Set Metrics:")
        print("-" * 40)
        for metric_name, metric_value in results["test_metrics"].items():
            print(f"  {metric_name:.<30} {metric_value:.4f}")
        print("\nMLflow:")
        print(f"  Experiment: {job_name}_tuning")
        print("  Study logged successfully")
        print(f"  Total runs: {len(results['study'].trials) + 1}")
        print("\nLogs saved to: logs/tuning.log")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nHyperparameter tuning failed. Check logs/tuning.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
