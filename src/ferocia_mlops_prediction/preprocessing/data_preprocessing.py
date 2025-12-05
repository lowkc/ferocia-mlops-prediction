"""Main data preprocessing pipeline for binary classification."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ferocia_mlops_prediction.preprocessing.config import DataConfig, PreprocessingMetadata


class DataLoader:
    """Handles loading raw data from CSV files."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize DataLoader.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

    def load_data(self, file_path: Path) -> pd.DataFrame:
        """Load raw data from CSV file.

        Args:
            file_path: Path to the CSV file.

        Returns:
            DataFrame containing the raw data.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the file is empty.
        """
        self.logger.info(f"Loading data from {file_path}")

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            # Load CSV with semicolon delimiter as per EDA notebook
            df = pd.read_csv(file_path, delimiter=";")
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            self.logger.debug(f"Columns: {df.columns.tolist()}")
            return df
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"Empty data file: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise


class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize DataCleaner.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

    def transform_pdays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform pdays feature into two features.

        Based on EDA, pdays has value -1 for customers not previously contacted.
        Split into:
        - previous_contact: Binary flag (0 if not contacted, 1 if contacted)
        - days_since_last_contact: Days since contact (0 if not contacted, otherwise pdays value)

        Args:
            df: Input DataFrame with pdays column.

        Returns:
            DataFrame with pdays replaced by two new features.
        """
        self.logger.info("Transforming pdays feature")

        if "pdays" not in df.columns:
            self.logger.warning("pdays column not found, skipping transformation")
            return df

        # Create binary flag for previous contact
        df["previous_contact"] = df["pdays"].apply(lambda x: 0 if x == -1 else 1)

        # Create days since last contact
        df["days_since_last_contact"] = df["pdays"].apply(lambda x: x if x != -1 else 0)

        # Drop original pdays column
        df = df.drop("pdays", axis=1)

        self.logger.info("Successfully transformed pdays into previous_contact and days_since_last_contact")
        return df

    def check_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for and report missing values.

        Args:
            df: Input DataFrame.

        Returns:
            Input DataFrame (unchanged).
        """
        self.logger.info("Checking for missing values")
        missing_counts = df.isna().sum()
        total_missing = missing_counts.sum()

        if total_missing > 0:
            self.logger.warning(f"Found {total_missing} missing values")
            for col, count in missing_counts[missing_counts > 0].items():
                self.logger.warning(f"  {col}: {count} missing values")
        else:
            self.logger.info("No missing values found")

        return df


class FeatureEngineer:
    """Handles feature engineering and encoding operations."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize FeatureEngineer.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

    def encode_binary_features(
        self, df: pd.DataFrame, binary_columns: list[str]
    ) -> pd.DataFrame:
        """Encode binary features from yes/no to 1/0.

        Args:
            df: Input DataFrame.
            binary_columns: List of column names with binary yes/no values.

        Returns:
            DataFrame with encoded binary features.
        """
        self.logger.info(f"Encoding binary features: {binary_columns}")

        for col in binary_columns:
            if col in df.columns:
                df[col] = df[col].map({"yes": 1, "no": 0}).astype(int)
                self.logger.debug(f"Encoded {col}: yes->1, no->0")
            else:
                self.logger.warning(f"Binary column {col} not found in DataFrame")

        return df

    def encode_categorical_features(
        self, df: pd.DataFrame, categorical_columns: list[str]
    ) -> pd.DataFrame:
        """One-hot encode categorical features.

        Args:
            df: Input DataFrame.
            categorical_columns: List of column names to one-hot encode.

        Returns:
            DataFrame with one-hot encoded categorical features.
        """
        self.logger.info(f"One-hot encoding categorical features: {categorical_columns}")

        # Only encode columns that exist in the DataFrame
        existing_cols = [col for col in categorical_columns if col in df.columns]
        missing_cols = set(categorical_columns) - set(existing_cols)

        if missing_cols:
            self.logger.warning(f"Categorical columns not found: {missing_cols}")

        if not existing_cols:
            self.logger.warning("No categorical columns to encode")
            return df

        df = pd.get_dummies(
            df,
            columns=existing_cols,
            prefix=existing_cols,
            drop_first=True,
            dtype=int,
        )

        self.logger.info(f"Created {len(df.columns)} columns after one-hot encoding")
        return df


class DataSplitter:
    """Handles train/test splitting of data."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize DataSplitter.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float,
        random_seed: int,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.

        Args:
            df: Input DataFrame.
            target_column: Name of the target variable column.
            test_size: Proportion of data to use for test set.
            random_seed: Random seed for reproducibility.
            stratify: Whether to use stratified splitting.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: If target column not found.
        """
        self.logger.info(f"Splitting data: test_size={test_size}, stratify={stratify}")

        if target_column not in df.columns:
            self.logger.error(f"Target column '{target_column}' not found")
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Perform train/test split
        stratify_arg = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify_arg
        )

        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        self.logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        self.logger.info(f"Test target distribution:\n{y_test.value_counts()}")

        return X_train, X_test, y_train, y_test


class PreprocessingPipeline:
    """Orchestrates the complete data preprocessing pipeline."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize preprocessing pipeline.

        Args:
            config: Configuration object for the pipeline.
        """
        self.config = config
        self.metadata = PreprocessingMetadata(
            test_size=config.test_size, random_seed=config.random_seed
        )

        # Setup logging
        self.logger = self._setup_logger()

        # Initialize components
        self.data_loader = DataLoader(self.logger)
        self.data_cleaner = DataCleaner(self.logger)
        self.feature_engineer = FeatureEngineer(self.logger)
        self.data_splitter = DataSplitter(self.logger)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger("preprocessing_pipeline")
        logger.setLevel(self.config.log_level)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.config.log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "preprocessing.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

        return logger

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Execute the complete preprocessing pipeline.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting data preprocessing pipeline")
            self.logger.info("=" * 80)

            # Step 1: Load data
            df = self.data_loader.load_data(self.config.raw_data_path)
            self.metadata.original_columns = df.columns.tolist()

            # Step 2: Check for missing values
            df = self.data_cleaner.check_missing_values(df)

            # Step 3: Transform pdays feature
            df = self.data_cleaner.transform_pdays(df)
            self.metadata.engineered_features = ["previous_contact", "days_since_last_contact"]

            # Step 4: Encode binary features
            binary_columns = ["default", "housing", "loan", "y"]
            self.metadata.binary_columns = binary_columns
            df = self.feature_engineer.encode_binary_features(df, binary_columns)

            # Step 5: Encode categorical features
            categorical_columns = ["job", "marital", "education", "contact", "month", "poutcome"]
            self.metadata.categorical_columns = categorical_columns
            df = self.feature_engineer.encode_categorical_features(df, categorical_columns)

            self.metadata.processed_columns = df.columns.tolist()

            # Step 6: Split data
            X_train, X_test, y_train, y_test = self.data_splitter.split_data(
                df,
                target_column="y",
                test_size=self.config.test_size,
                random_seed=self.config.random_seed,
                stratify=self.config.stratify,
            )

            self.metadata.train_samples = len(X_train)
            self.metadata.test_samples = len(X_test)

            # Step 7: Save processed data
            self._save_processed_data(X_train, X_test, y_train, y_test)

            # Step 8: Save metadata
            if self.config.save_metadata:
                self._save_metadata()

            self.logger.info("=" * 80)
            self.logger.info("Preprocessing pipeline completed successfully")
            self.logger.info("=" * 80)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Save processed datasets to CSV files.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
        """
        self.logger.info(f"Saving processed data to {self.config.output_dir}")

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Combine features and target for saving
        train_data = X_train.copy()
        train_data["y"] = y_train.values

        test_data = X_test.copy()
        test_data["y"] = y_test.values

        # Save to CSV
        train_path = self.config.output_dir / "train.csv"
        test_path = self.config.output_dir / "test.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        self.logger.info(f"Saved training data to {train_path}")
        self.logger.info(f"Saved test data to {test_path}")

    def _save_metadata(self) -> None:
        """Save preprocessing metadata to JSON file."""
        metadata_path = self.config.output_dir / "preprocessing_metadata.json"

        metadata_dict: dict[str, Any] = {
            "original_columns": self.metadata.original_columns,
            "processed_columns": self.metadata.processed_columns,
            "binary_columns": self.metadata.binary_columns,
            "categorical_columns": self.metadata.categorical_columns,
            "engineered_features": self.metadata.engineered_features,
            "target_column": self.metadata.target_column,
            "train_samples": self.metadata.train_samples,
            "test_samples": self.metadata.test_samples,
            "test_size": self.metadata.test_size,
            "random_seed": self.metadata.random_seed,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        self.logger.info(f"Saved preprocessing metadata to {metadata_path}")
