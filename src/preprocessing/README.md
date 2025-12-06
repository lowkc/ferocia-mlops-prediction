# Data Preprocessing Pipeline

A production-ready data pipeline for preprocessing data used for binary classification of whether or not a customer will subscribe to a term deposite.

## Overview

This module a preprocessing pipeline that transforms raw banking marketing data a dataset ready for ingestion into a binary classification model pipeline. The pipeline uses a YAML-based configuration file for reproducibility.

## Features

- **Modular Design**: Separate classes for loading, cleaning, feature engineering, and splitting
- **YAML Configuration**: Externalized configuration files for easy parameter management
- **Logging**: Comprehensive logging to both console and file
- **Dataclass-based Config**: Structured configuration with validation using Python dataclasses
- **Reproducibility**: Random seed support for deterministic results
- **Metadata Tracking**: Automatic saving of preprocessing metadata for reproducibility

## Architecture

### Components

1. **DataConfig**: Configuration dataclass for data paths and splitting parameters
2. **PreprocessingConfig**: Configuration dataclass for preprocessing operations
3. **PreprocessingMetadata**: Tracks metadata about preprocessing operations
4. **DataLoader**: Loads raw CSV data with proper delimiter handling (semicolon-delimited)
5. **DataCleaner**: Handles missing values (imputation with mean) and duplicate removal
6. **FeatureEngineer**: Creates new features (pdays transformation, total_contacts)
7. **DataSplitter**: Stratified train/test splitting with configurable ratios
8. **PreprocessingPipeline**: Orchestrates the complete pipeline with YAML configuration support

### Data Transformations

The pipeline performs the following transformations (configurable via YAML):

1. **Missing Value Handling**: Imputes missing numeric values with mean (when `handle_missing: true`). Note: mean is used as an example, please see comments within the code.

2. **Duplicate Removal**: Drops duplicate rows (when `drop_duplicates: true`)

3. **Feature Engineering** (when `engineer_features: true`):
   - **pdays Transformation**: Splits into two features:
     - `previous_contact`: Binary flag (0=not contacted, 1=contacted)
     - `days_since_last_contact`: Days since last contact (0 if not contacted)
   - **total_contacts**: Sum of `campaign` + `previous` to capture total contact history

Note: The current implementation focuses on feature engineering. Encoding operations (binary and one-hot encoding) are not included as preprocessing steps. These steps are left to the modelling pipeline for simplicity and flexibility (e.g., ability to add new groups to columns such as `job`).

## Usage

### Command Line

Run the preprocessing pipeline with default configuration (uses `confs/preprocess.yaml`):

```bash
uv run python -m preprocessing.run_preprocessing
```

With a custom configuration file:

```bash
uv run python -m preprocessing.run_preprocessing --config path/to/config.yaml
```

Override log level:

```bash
uv run python -m preprocessing.run_preprocessing --log-level DEBUG
```

### Python API

```python
from pathlib import Path
from preprocessing.config import DataConfig, PreprocessingConfig, load_config
from preprocessing.data_preprocessing import PreprocessingPipeline

# Option 1: Load configuration from YAML file
data_config, preprocessing_config = load_config("confs/preprocess.yaml")

# Option 2: Create configuration programmatically
data_config = DataConfig(
    raw_data_path=Path("data/raw/dataset.csv"),
    output_dir=Path("data/processed"),
    target_column="y",
    test_size=0.2,
    random_seed=42,
    stratify=True,
    log_level="INFO"
)

preprocessing_config = PreprocessingConfig(
    handle_missing=True,
    drop_duplicates=True,
    engineer_features=True,
    save_metadata=True
)

# Run pipeline
pipeline = PreprocessingPipeline(data_config, preprocessing_config)
X_train, X_test, y_train, y_test = pipeline.run()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.columns.tolist()}")
```

## Configuration

### YAML Configuration File

The pipeline uses a YAML configuration file (default: `confs/preprocess.yaml`) with two main sections:

```yaml
data:
  raw_path: "data/raw/dataset.csv"
  processed_dir: "data/processed"
  target_column: "y"
  test_size: 0.2
  random_state: 42
  stratify: true

preprocessing:
  handle_missing: true
  drop_duplicates: true
  engineer_features: true
  save_metadata: true
```

### Configuration Parameters

#### DataConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_path` | str | `data/raw/dataset.csv` | Path to raw CSV file |
| `processed_dir` | str | `data/processed` | Output directory for processed files |
| `target_column` | str | `y` | Name of the target variable column |
| `test_size` | float | 0.2 | Proportion for test set (0.0-1.0) |
| `random_state` | int | 42 | Random seed for reproducibility |
| `stratify` | bool | true | Use stratified splitting |

#### PreprocessingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handle_missing` | bool | true | Impute missing values with mean |
| `drop_duplicates` | bool | true | Remove duplicate rows |
| `engineer_features` | bool | true | Perform feature engineering |
| `save_metadata` | bool | true | Save preprocessing metadata |

## Output Files

The pipeline creates the following files in the output directory:

1. **train.csv**: Training dataset with features and target
2. **test.csv**: Test dataset with features and target
3. **preprocessing_metadata.json**: Metadata about the preprocessing operations (if `save_metadata: true`)

### Metadata Structure

```json
{
  "original_columns": ["age", "job", "marital", "education", ...],
  "columns_after_processing": ["age", "job", "marital", "education", ..., "previous_contact", "days_since_last_contact", "total_contacts"],
  "engineered_features": ["previous_contact", "days_since_last_contact", "total_contacts"],
  "target_column": "y",
  "train_samples": 36168,
  "test_samples": 9043,
  "test_size": 0.2,
  "random_seed": 42
}
```


## Logging

Logs are written to:
- **Console**: INFO level and above
- **File**: `logs/preprocessing.log` (all levels including DEBUG)

Log format:
```
2025-12-05 17:35:06,669 - preprocessing_pipeline - INFO - Starting data preprocessing pipeline
```

## Testing

Run the test suite:

```bash
# Run all preprocessing tests
uv run pytest tests/test_preprocessing.py -v

# Run with coverage
uv run pytest tests/test_preprocessing.py --cov=src/ferocia_mlops_prediction/preprocessing
```

## Error Handling

The pipeline includes comprehensive error handling:

- **FileNotFoundError**: If raw data file doesn't exist
- **ValueError**: For invalid configuration parameters or missing target column
- **EmptyDataError**: If CSV file is empty
- Generic exception handling with full stack traces in logs

## Dependencies

- `pandas>=2.2.0`: Data manipulation and CSV I/O
- `scikit-learn>=1.6.0`: Train/test splitting with stratification
- `numpy>=2.2.0`: Numerical operations (transitive dependency)
- `pyyaml>=6.0.0`: YAML configuration file parsing