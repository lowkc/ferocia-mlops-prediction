# Data Preprocessing Pipeline

A production-ready data preprocessing pipeline for binary classification, built with MLOps and software engineering best practices.

## Overview

This module provides a modular, type-safe, and well-tested preprocessing pipeline that transforms raw banking marketing data into clean, encoded datasets ready for binary classification models.

## Features

- **Modular Design**: Separate classes for loading, cleaning, feature engineering, and splitting
- **Type Safety**: Full type hints throughout the codebase
- **Logging**: Comprehensive logging to both console and file
- **Configuration**: Dataclass-based configuration management
- **Error Handling**: Robust error handling with informative messages
- **Reproducibility**: Random seed support for deterministic results
- **Metadata Tracking**: Automatic saving of preprocessing metadata for reproducibility
- **Testing**: Comprehensive unit tests with high coverage

## Architecture

### Components

1. **DataConfig**: Configuration dataclass with validation
2. **DataLoader**: Loads raw CSV data with proper delimiter handling
3. **DataCleaner**: Handles missing values and feature transformations
4. **FeatureEngineer**: Binary and one-hot encoding of features
5. **DataSplitter**: Stratified train/test splitting
6. **PreprocessingPipeline**: Orchestrates the complete pipeline

### Data Transformations

Based on the EDA notebook analysis, the pipeline performs:

1. **pdays Transformation**: Splits into two features:
   - `previous_contact`: Binary flag (0=not contacted, 1=contacted)
   - `days_since_last_contact`: Days since last contact (0 if not contacted)

2. **Binary Encoding**: Converts yes/no to 1/0 for:
   - `default`, `housing`, `loan`, `y` (target)

3. **One-Hot Encoding**: Creates dummy variables with drop_first=True for:
   - `job`, `marital`, `education`, `contact`, `month`, `poutcome`

## Usage

### Command Line

Run the preprocessing pipeline with default settings:

```bash
uv run python -m ferocia_mlops_prediction.preprocessing.run_preprocessing
```

With custom parameters:

```bash
uv run python -m ferocia_mlops_prediction.preprocessing.run_preprocessing \
    --raw-data-path data/dataset.csv \
    --output-dir data/processed \
    --test-size 0.2 \
    --random-seed 42 \
    --log-level INFO
```

### Python API

```python
from pathlib import Path
from ferocia_mlops_prediction.preprocessing import DataConfig, PreprocessingPipeline

# Create configuration
config = DataConfig(
    raw_data_path=Path("data/dataset.csv"),
    output_dir=Path("data/processed"),
    test_size=0.2,
    random_seed=42,
    stratify=True,
    log_level="INFO",
    save_metadata=True
)

# Run pipeline
pipeline = PreprocessingPipeline(config)
X_train, X_test, y_train, y_test = pipeline.run()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.columns.tolist()}")
```

## Output Files

The pipeline creates the following files in the output directory:

1. **train.csv**: Training dataset with features and target
2. **test.csv**: Test dataset with features and target
3. **preprocessing_metadata.json**: Metadata about the preprocessing operations

### Metadata Structure

```json
{
  "original_columns": ["age", "job", ...],
  "processed_columns": ["age", "balance", "job_admin", ...],
  "binary_columns": ["default", "housing", "loan", "y"],
  "categorical_columns": ["job", "marital", "education", "contact", "month", "poutcome"],
  "engineered_features": ["previous_contact", "days_since_last_contact"],
  "target_column": "y",
  "train_samples": 36168,
  "test_samples": 9043,
  "test_size": 0.2,
  "random_seed": 42
}
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `raw_data_path` | Path | `data/dataset.csv` | Path to raw CSV file |
| `output_dir` | Path | `data/processed` | Output directory for processed files |
| `test_size` | float | 0.2 | Proportion for test set (0.0-1.0) |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `stratify` | bool | True | Use stratified splitting |
| `log_level` | str | INFO | Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `save_metadata` | bool | True | Save preprocessing metadata |

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

## Example Output

```
================================================================================
Starting data preprocessing pipeline
================================================================================
Loading data from data/dataset.csv
Successfully loaded 45211 rows and 17 columns
Checking for missing values
No missing values found
Transforming pdays feature
Successfully transformed pdays into previous_contact and days_since_last_contact
Encoding binary features: ['default', 'housing', 'loan', 'y']
One-hot encoding categorical features: ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
Created 44 columns after one-hot encoding
Splitting data: test_size=0.2, stratify=True
Train set: 36168 samples
Test set: 9043 samples
Saving processed data to data/processed
================================================================================
Preprocessing pipeline completed successfully
================================================================================

Processed data saved to: data/processed
Training samples: 36168
Test samples: 9043
Number of features: 43
```

## Best Practices Implemented

1. **Type Hints**: All functions and methods have complete type annotations
2. **Dataclasses**: Used for structured configuration and metadata
3. **Logging**: Comprehensive logging at appropriate levels
4. **Modularity**: Separate classes for each responsibility
5. **Documentation**: Google-style docstrings for all public APIs
6. **Testing**: Unit tests for all components with mocking
7. **Error Handling**: Try-except blocks with informative messages
8. **Reproducibility**: Random seed support and metadata tracking
9. **Validation**: Input validation in configuration
10. **Clean Code**: PEP 8 compliant, properly formatted with ruff

## Dependencies

- `pandas>=2.2.0`: Data manipulation
- `scikit-learn>=1.6.0`: Train/test splitting
- `numpy>=2.2.0`: Numerical operations

## Future Enhancements

Potential improvements for production use:

1. **Feature Scaling**: Add standardization/normalization options
2. **Data Validation**: Add schema validation with pandera or pydantic
3. **Parallel Processing**: Add support for large datasets
4. **MLflow Integration**: Track preprocessing as MLflow experiments
5. **Data Quality Checks**: Add automated data quality assertions
6. **Feature Selection**: Add automated feature selection methods
7. **Pipeline Serialization**: Save fitted transformers for inference
