"""Command-line script to run the data preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

from ferocia_mlops_prediction.preprocessing.config import DataConfig
from ferocia_mlops_prediction.preprocessing.data_preprocessing import PreprocessingPipeline


def main() -> int:
    """Main entry point for preprocessing script.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Run data preprocessing pipeline for binary classification"
    )
    parser.add_argument(
        "--raw-data-path",
        type=Path,
        default=Path("data/dataset.csv"),
        help="Path to raw dataset CSV file (default: data/dataset.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't save preprocessing metadata",
    )

    args = parser.parse_args()

    # Create configuration
    config = DataConfig(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.random_seed,
        stratify=not args.no_stratify,
        log_level=args.log_level,
        save_metadata=not args.no_metadata,
    )

    # Run pipeline
    try:
        pipeline = PreprocessingPipeline(config)
        X_train, X_test, y_train, y_test = pipeline.run()

        print("\n" + "=" * 80)
        print("Preprocessing completed successfully!")
        print("=" * 80)
        print(f"\nProcessed data saved to: {config.output_dir}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {len(X_train.columns)}")
        print(f"\nLogs saved to: logs/preprocessing.log")

        if config.save_metadata:
            print(f"Metadata saved to: {config.output_dir / 'preprocessing_metadata.json'}")

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print("\nPreprocessing failed. Check logs/preprocessing.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
