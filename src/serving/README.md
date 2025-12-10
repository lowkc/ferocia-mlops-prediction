# Model Serving API

This directory contains the FastAPI application for serving the trained term deposit prediction model.

## Architecture

The serving solution consists of three main components:

1. **`predictor.py`**: `ModelPredictor` class that handles model loading and prediction
2. **`api.py`**: FastAPI application with REST endpoints
3. **`tools/download_mlflow_model.py`**: Script to download model from MLflow tracking server

## Setup

### 1. Download the Model

First, download the trained model from MLflow:

```bash
python tools/download_mlflow_model.py --config confs/deployment.yaml
```

This will download the best model from MLflow and save it to `models/model.pkl`.

### 2. Install Dependencies

Ensure FastAPI and related dependencies are installed:

```bash
uv sync
```

### 3. Run the API Locally

Start the FastAPI server:

```bash
python -m src.serving.api
```

Or using uvicorn directly:

```bash
uvicorn src.serving.api:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Health Check
```bash
GET /health
```

Returns the health status of the service and whether the model is loaded.

### Model Info
```bash
GET /model-info
```

Returns information about the loaded model.

### Predict
```bash
POST /predict
```

Make a prediction on input data. Accepts JSON with the following schema:

**Required Fields:**

*Numerical Features:*
- `age`: int (18-100) - Age of the client
- `balance`: int - Average yearly balance in euros
- `day`: int (1-31) - Last contact day of the month
- `duration`: int (≥0) - Last contact duration in seconds
- `campaign`: int (≥0) - Number of contacts performed during this campaign
- `previous`: int (≥0) - Number of contacts performed before this campaign
- `total_contacts`: int (≥0) - Total number of contacts (campaign + previous)
- `days_since_last_contact`: int (≥0) - Number of days since client was last contacted (0 if not contacted)

*Binary Features:*
- `previous_contact`: int (0 or 1) - Previous contact flag
- `default`: string - Has credit in default?
- `housing`: string - Has housing loan?
- `loan`: string - Has personal loan?

*Categorical Features:*
- `job`: string - Type of job
- `marital`: string - Marital status
- `education`: string - Education level
- `contact`: string - Contact communication type
- `month`: string - Last contact month of year
- `poutcome`: string - Outcome of the previous marketing campaign

**Example Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

Or inline:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 56,
    "balance": 1234,
    "day": 5,
    "duration": 261,
    "campaign": 1,
    "previous": 0,
    "total_contacts": 1,
    "days_since_last_contact": 0,
    "previous_contact": 0,
    "default": "no",
    "housing": "no",
    "loan": "no",
    "job": "housemaid",
    "marital": "married",
    "education": "basic.4y",
    "contact": "telephone",
    "month": "may",
    "poutcome": "nonexistent"
  }'
```

**Example Response:**
```json
{
  "prediction": 0,
  "probability": 0.1234,
  "probabilities": {
    "class_0": 0.8766,
    "class_1": 0.1234
  }
}
```

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Deployment

### Build the Docker Image

```bash
# First, build the wheel
just package

# Then build the Docker image
docker build -t term-deposit-prediction-api .
```

### Run the Container

```bash
docker run -p 8000:8000 term-deposit-prediction-api
```

The API will be accessible at `http://localhost:8000`.

## Configuration

Model serving configuration is defined in `confs/deployment.yaml`:

```yaml
model:
  name: "term_deposit_prediction_tuning_best_model"
  local_storage_path: "models/"

api:
  host: "127.0.0.1"
  port: 8000
  workers: 1
  log_level: "info"
```

## Preprocessing

The model includes a scikit-learn Pipeline that automatically handles:
- One-hot encoding of categorical features
- Standard scaling of numerical features
- Feature engineering (pdays transformation, total_contacts calculation)

Input data should be provided in raw format matching the original dataset schema. The preprocessing is applied automatically by the model pipeline.

## Error Handling

The API includes comprehensive error handling:
- **400 Bad Request**: Invalid input data
- **500 Internal Server Error**: Prediction failure
- **503 Service Unavailable**: Model not loaded

All errors return JSON with a `detail` field explaining the issue.

## Logging

Logs are written to stdout and include:
- Model loading status
- Prediction requests and results
- Error details for debugging

## Testing

Test the API using the provided example:

```bash
# Start the API
python -m src.serving.api

# In another terminal, make a test request
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_request.json
```
