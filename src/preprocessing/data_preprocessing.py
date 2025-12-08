"""Main data preprocessing pipeline for binary classification."""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from entities.configs import PreprocessingDataConfig, PreprocessingMetadata, PreprocessingConfig


class DataLoader:
    """Handles loading raw data from CSV files."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialise DataLoader.

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
            df = pd.read_csv(file_path, delimiter=";")
            self.logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            self.logger.debug(f"Columns: {df.columns.tolist()}")
            return df
        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty data file: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise


class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialise DataCleaner.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

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

    def impute_missing_values_with_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame with the mean.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values imputed.
        """
        # We use mean as an example here, but other strategies could be applied such as mode or median.
        # Ideally the function would be more flexible to take in different strategies per column.
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                mean_value = df[col].mean()
                df.fillna({col: mean_value}, inplace=True)
                self.logger.info(f"Imputed missing values in {col} with mean: {mean_value}")

        return df

    def drop_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows from the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Input DataFrame with duplicate rows removed.
        """
        duplicate_rows = df.duplicated()
        if duplicate_rows.sum():
            self.logger.info(f"Dropping {duplicate_rows.sum()} duplicate rows from data")
            df = df.drop_duplicates()
        return df


class FeatureEngineer:
    """Handles feature engineering and encoding operations."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialise FeatureEngineer.

        Args:
            logger: Logger instance for tracking operations.
        """
        self.logger = logger

    def transform_pdays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform pdays feature into two features.

        pdays is the number of days that passed since last contact and has value -1 for customers not
        previously contacted. We split this into two features to separate the information into a binary
        value and a continuous value:
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

        self.logger.info(
            "Successfully transformed pdays into previous_contact and days_since_last_contact"
        )
        return df

    def total_contacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create total_contacts feature as sum of all previous contacts.

        Args:
            df: Input DataFrame with "campaign" and "previous" columns.

        Returns:
            DataFrame with additional "total contacts" column (numeric).
        """
        self.logger.info("Creating total_contacts feature")

        if "campaign" not in df.columns or "previous" not in df.columns:
            self.logger.warning(
                "campaign or previous column not found, skipping total_contacts feature"
            )
            return df

        df["total_contacts"] = df["campaign"] + df["previous"]

        self.logger.info("Successfully created total_contacts feature")
        return df

    # Any additional features are engineered here. In the interest of time and scope, feature
    # engineering has been limited to 2 features only.


class DataSplitter:
    """Handles train/test splitting of data."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialise DataSplitter.

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
            Tuple of (x_train, x_test, y_train, y_test).

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
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=stratify_arg
        )

        self.logger.info(f"Train set: {len(x_train)} samples")
        self.logger.info(f"Test set: {len(x_test)} samples")
        self.logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        self.logger.info(f"Test target distribution:\n{y_test.value_counts()}")

        return x_train, x_test, y_train, y_test


class PreprocessingPipeline:
    """Orchestrates the complete data preprocessing pipeline."""

    def __init__(
        self, data_config: PreprocessingDataConfig, preprocessing_config: PreprocessingConfig
    ) -> None:
        """Initialise preprocessing pipeline.

        Args:
            config: Configuration object for the pipeline.
        """
        self.data_config = data_config
        self.preprocessing_config = preprocessing_config
        self.metadata = PreprocessingMetadata()

        # Setup logging
        self.logger = self._setup_logger()

        # Initialise components
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
        logger.setLevel(self.data_config.log_level)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.data_config.log_level)
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
            Tuple of (x_train, x_test, y_train, y_test).

        Raises:
            Exception: If any step in the pipeline fails.
        """
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting data preprocessing pipeline")
            self.logger.info("=" * 80)

            # Step 1: Load data
            df = self.data_loader.load_data(self.data_config.raw_data_path)
            self.metadata.original_columns = df.columns.tolist()

            # Step 2: Validate and clean data
            df = self.data_cleaner.check_missing_values(df)
            if self.preprocessing_config.handle_missing:
                df = self.data_cleaner.impute_missing_values_with_mean(df)
            if self.preprocessing_config.drop_duplicates:
                df = self.data_cleaner.drop_duplicate_rows(df)

            # Step 3: Feature engineering
            if self.preprocessing_config.engineer_features:
                self.logger.info("Starting feature engineering")
                df = self.feature_engineer.transform_pdays(df)
                self.metadata.engineered_features = ["previous_contact", "days_since_last_contact"]
                df = self.feature_engineer.total_contacts(df)
                self.metadata.engineered_features.append("total_contacts")
                self.logger.info(
                    f"Completed feature engineering, added features: {self.metadata.engineered_features}"
                )

            self.metadata.columns_after_processing = df.columns.tolist()

            # Step 6: Split data
            x_train, x_test, y_train, y_test = self.data_splitter.split_data(
                df,
                target_column="y",
                test_size=self.data_config.test_size,
                random_seed=self.data_config.random_seed,
                stratify=self.data_config.stratify,
            )

            self.metadata.train_samples = len(x_train)
            self.metadata.test_samples = len(x_test)

            # Step 7: Save processed data
            self._save_processed_data(x_train, x_test, y_train, y_test)

            # Step 8: Save metadata
            if self.preprocessing_config.save_metadata:
                self._save_metadata()

            self.logger.info("=" * 80)
            self.logger.info("Preprocessing pipeline completed successfully")
            self.logger.info("=" * 80)

            return x_train, x_test, y_train, y_test

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _save_processed_data(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """Save processed datasets to CSV files.

        Args:
            x_train: Training features.
            x_test: Test features.
            y_train: Training target.
            y_test: Test target.
        """
        self.logger.info(f"Saving processed data to {self.data_config.output_dir}")

        # Create output directory
        self.data_config.output_dir.mkdir(parents=True, exist_ok=True)

        # Combine features and target for saving
        train_data = x_train.copy()
        train_data["y"] = y_train.values

        test_data = x_test.copy()
        test_data["y"] = y_test.values

        # Save to CSV
        train_path = self.data_config.output_dir / "train.csv"
        test_path = self.data_config.output_dir / "test.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        self.logger.info(f"Saved training data to {train_path}")
        self.logger.info(f"Saved test data to {test_path}")

    def _save_metadata(self) -> None:
        """Save preprocessing metadata to JSON file."""
        metadata_path = self.data_config.output_dir / "preprocessing_metadata.json"

        metadata_dict: dict[str, Any] = {
            "original_columns": self.metadata.original_columns,
            "columns_after_processing": self.metadata.columns_after_processing,
            "engineered_features": self.metadata.engineered_features,
            "target_column": self.data_config.target_column,
            "train_samples": self.metadata.train_samples,
            "test_samples": self.metadata.test_samples,
            "test_size": self.data_config.test_size,
            "random_seed": self.data_config.random_seed,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        self.logger.info(f"Saved preprocessing metadata to {metadata_path}")
