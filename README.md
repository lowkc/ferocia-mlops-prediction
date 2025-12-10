# Term Deposit Subscription Prediction

An end-to-end MLOps pipeline for predicting whether a customer will subscribe to a term deposit.

## Project Overview

This binary classification system predicts customer subscription to term deposits based on banking and demographic features. The project implements a full MLOps lifecycle:

- **Data Pipeline**: Automated data validation and feature engineering
- **Training Pipeline**: Model training with experiment tracking
- **Hyperparameter Tuning**: optimisation using Optuna
- **Model Registry**: MLflow-based model versioning and management
- **Deployment**: Production-ready FastAPI service

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install Dependencies

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

This will create a virtual environment and install all required packages including:
- scikit-learn, xgboost, pandas, numpy (ML stack)
- mlflow (experiment tracking)
- optuna (hyperparameter tuning)
- fastapi, uvicorn (API serving)
- pytest, pytest-cov (testing)

## Quick Start
This project uses YAML files stored in `confs/` to handle all pipeline configurations.

### 1. Data Preprocessing
Define data test/train split size and preprocessing in `confs/preprocess.yaml`

```bash
# Preprocess data with default configuration
uv run python src/run_preprocessing.py

# Or with custom config
uv run python src/run_preprocessing.py --config confs/preprocess.yaml
```

### 2. Model Training
Define model type, parameters, features, and train/test split in `confs/training.yaml`

```bash
# Train model and log to MLflow
uv run python src/run_training.py

# Training will automatically:
# - Load preprocessed data
# - Train XGBoost model
# - Evaluate on test set
# - Log metrics and model to MLflow
```

### 3. Hyperparameter Tuning
Hyperparameter tuning uses Optuna to perform multiple studies to find the optimal set of parameters. Define parameters, search spaces, etc. using `confs/tuning.yaml`
```bash
# Run hyperparameter optimization
uv run python src/run_tuning.py

# Configure tuning in confs/tuning.yaml:
# - Number of trials
# - Parameter search space
# - Optimization metric
```

### 4. Deploy Model
A FastAPI application is uesd for serving the trained model. Define the model name and local output directory using using `confs/deployment.yaml`, and the version of the model to deploy in the runscript below.
```bash
# Download model from MLflow
uv run python tools/download_mlflow_model.py --config confs/deployment.yaml --version latest

# Start API server
uv run uvicorn src.serving.api:app --host 127.0.0.1 --port 8000

# Or use Docker
docker build -t mlops-prediction-api .
docker run -p 8000:8000 mlops-prediction-api
```

### 5. Make Predictions

```bash
# Check API health
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## Detailed Usage

### Data Preprocessing

The preprocessing pipeline handles:
- **Validation**: Checks for missing values and data types
- **Feature Engineering**:
  - One-hot encoding for categorical variables
  - Age binning
  - Balance categorization
- **Output**: Clean, preprocessed data ready for training

Configuration in `confs/preprocess.yaml`:
```yaml
input_path: data/raw/bank.csv
output_path: preprocessing/preprocessed_data.csv
validate_data: true
```

### Model Training

Training supports multiple configurations:

```yaml
# confs/training.yaml
model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

train_test_split:
  test_size: 0.2
  random_state: 42

cross_validation:
  enabled: true
  n_splits: 5
```

Features:
- Stratified train-test split
- K-fold cross-validation
- Comprehensive metrics logging
- Model persistence in MLflow

### Hyperparameter Tuning

Optuna-based tuning optimizes:
- `n_estimators`: Number of boosting rounds
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `min_child_weight`: Minimum sum of instance weight
- `subsample`: Subsample ratio of training instances
- `colsample_bytree`: Subsample ratio of columns

```yaml
# confs/tuning.yaml
n_trials: 50
timeout_seconds: 3600
study_name: bank_marketing_optimization
```

### Model Deployment

The FastAPI service provides:

#### Endpoints

**`POST /predict`** - Make predictions
```json
{
  "age": 35,
  "job": "management",
  "marital": "married",
  "education": "tertiary",
  "default": 0,
  "balance": 1500,
  "housing": 1,
  "loan": 0,
  "contact": "cellular",
  "day": 15,
  "month": "may",
  "duration": 300,
  "campaign": 2,
  "pdays": -1,
  "previous_contact": 0,
  "poutcome": "unknown"
}
```

**`GET /health`** - Health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-01-10T12:00:00"
}
```

**`GET /model/info`** - Model metadata
```json
{
  "model_type": "XGBClassifier",
  "version": "1",
  "features": [...],
  "loaded_at": "2025-01-10T12:00:00"
}
```

**`GET /`** - Root endpoint with API information

## Configuration

All pipelines are configured via YAML files in `confs/`:

### `preprocess.yaml`
```yaml
input_path: data/raw/bank.csv
output_path: preprocessing/preprocessed_data.csv
validate_data: true
log_level: INFO
```

### `training.yaml`
```yaml
data:
  preprocessed_path: preprocessing/preprocessed_data.csv

model:
  type: xgboost
  params:
    n_estimators: 100
    max_depth: 6

mlflow:
  experiment_name: bank_marketing_classification
  tracking_uri: sqlite:///mlflow.db
```

### `tuning.yaml`
```yaml
data:
  preprocessed_path: preprocessing/preprocessed_data.csv

tuning:
  n_trials: 50
  timeout_seconds: 3600

search_space:
  n_estimators: [50, 300]
  max_depth: [3, 10]
  learning_rate: [0.01, 0.3]
```

### `deployment.yaml`
```yaml
model:
  name: bank_marketing_model
  local_storage_path: models/
  version: latest

api:
  host: 0.0.0.0
  port: 8000
  reload: false
```

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_api.py -v

# Run specific test
uv run pytest tests/test_api.py::TestAPIEndpoints::test_predict_success -v
```

Test Coverage:
- **API Tests** (27 tests): Endpoint testing, validation, error handling
- **Predictor Tests** (19 tests): Model loading, prediction logic
- **Preprocessing Tests**: Data validation, transformation
- **Training Tests**: Model training, evaluation
- **Tuning Tests**: Hyperparameter optimization

Current coverage: **60+ tests passing**

Please note: the majority of unit tests were generated using AI in the interest of time. They have been tested for functionality.

## Development

### Code Quality

Pre-commit hooks ensure code quality:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files
```

Includes:
- Code formatting (black, isort)
- Linting (flake8, pylint)
- Type checking (mypy)
- Security checks (bandit)

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Implement changes**
   - Add code in appropriate `src/` directory
   - Update configuration if needed
   - Add tests in `tests/`

3. **Run tests**
   ```bash
   uv run pytest
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```

### Docker Development

```bash
# Build image
docker build -t mlops-prediction-api:dev .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models mlops-prediction-api:dev

# View logs
docker logs -f <container-id>
```

### MLflow UI

```bash
# Start MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# Access at http://localhost:5000
```

## Tool Stack

### AI Assistance
- **IDE:** VSCode
- **AI Coding Assistants:** Cline (Claude Sonnet 4.5) and GitHub Co-Pilot Inline Chat (Claude Haiku 4.5) were both used for code generation
- **Transcripts:** for all Cline chats are stored in the folder `ai_transcripts`

## Model Performance

Current model metrics (on test set):
- **Accuracy**: ~89%
- **Precision**: ~53%
- **Recall**: ~76%
- **F1 Score**: ~63%

Model performance could be improved through several ways:
- Exploring other tree and non-tree based model types such as LightGBM or CatBoost
- More sophisticated feature engineering
- Analysis of feature relevance to remove and irrelevant or unimportant features for the model
- Addressing class imbalance in the training dataset through methods such using SMOTE to synthesise more data for the positive class

However this is out of scope for the current work.
