"""Command-line script to run the data preprocessing pipeline."""

import argparse
import sys

from preprocessing.config import load_config
from preprocessing.data_preprocessing import PreprocessingPipeline


def main() -> int:
    """Main entry point for preprocessing script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(description="Preprocess data for ML model training")
    parser.add_argument(
        "--config",
        type=str,
        default="confs/preprocess.yaml",
        help="Path to configuration YAML file (default: confs/preprocess.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override logging level from config file",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        data_config, preprocessing_config = load_config(args.config)

        # Override log level if provided via CLI
        if args.log_level:
            data_config.log_level = args.log_level

    except Exception as e:
        print(f"\nError loading configuration: {e}", file=sys.stderr)
        return 1

    # Run pipeline
    try:
        pipeline = PreprocessingPipeline(data_config, preprocessing_config)
        X_train, X_test, y_train, y_test = pipeline.run()

        print("\n" + "=" * 80)
        print("Preprocessing completed successfully!")
        print("=" * 80)
        print(f"\nProcessed data saved to: {data_config.output_dir}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {len(X_train.columns)}")
        print("\nLogs saved to: logs/preprocessing.log")

        if preprocessing_config.save_metadata:
            print(f"Metadata saved to: {data_config.output_dir / 'preprocessing_metadata.json'}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nPreprocessing failed. Check logs/preprocessing.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
