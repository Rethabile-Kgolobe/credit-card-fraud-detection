# Credit Card Fraud Detection API

An end-to-end machine learning project that detects potentially fraudulent credit card transactions using an XGBoost classification model. The trained model is served through a FastAPI application, containerized with Docker, tested using GitHub Actions, and deployed on Render.

## Live Demo

- **API:** https://credit-card-fraud-detection-th5f.onrender.com/
- **Interactive API documentation:** https://credit-card-fraud-detection-th5f.onrender.com/docs

*Note: The Render free instance may take a little longer to respond after a period of inactivity.*

## Project Overview

Credit card fraud is a classification problem in which fraudulent transactions must be identified among a much larger number of legitimate transactions.

This project develops a machine learning model to classify transactions as either:

- **Legitimate (0)**
- **Fraudulent (1)**

The project covers the machine learning workflow and the deployment of the trained model as a REST API.

## Dataset

The dataset contains anonymized credit card transaction information.

- Target variable: `Class`
- Input features: `Time`, `V1`–`V28`, and `Amount`
- Total input features: 30
- Legitimate transactions: 284,315
- Fraudulent transactions: 492

The dataset is highly imbalanced, making fraud detection a challenging classification task.

The original dataset is not included in this repository because of its large file size.

## Machine Learning Approach

The exploratory analysis and model development were performed in a Jupyter notebook.

The workflow included:

1. Exploratory Data Analysis (EDA).
2. Examination of the class imbalance.
3. Stratified train-test splitting.
4. Baseline model development using Random Forest and XGBoost.
5. Hyperparameter optimization using Optuna.
6. Training and evaluation of the final XGBoost model.
7. Saving the trained model as `artifacts/model.pkl`.

The final model is loaded by the FastAPI application to generate predictions.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data analysis | Pandas, NumPy |
| Machine learning | XGBoost, Scikit-learn |
| Hyperparameter tuning | Optuna |
| API | FastAPI |
| API server | Uvicorn |
| Model persistence | Joblib |
| Containerization | Docker |
| Automated testing | Pytest |
| CI/CD | GitHub Actions |
| Deployment | Render |
| Version control | Git and GitHub |

## Project Structure

```text
ProjectPorfolio_1/
├── .github/
│   └── workflows/
│       └── ci.yml
├── app/
│   └── main.py
├── artifacts/
│   └── model.pkl
├── configs/
├── data/
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── EDA.ipynb
├── scripts/
├── src/
│   └── models/
│       └── predict.py
├── tests/
│   └── test_main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .dockerignore
└── train.py
```

## Running the API Locally

### 1. Clone the repository

```powershell
git clone https://github.com/Rethabile-Kgolobe/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the API

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

## Running with Docker

Build the Docker image from the project root:

```powershell
docker build -f docker/Dockerfile -t credit-card-fraud-api .
```

Run the container:

```powershell
docker run -p 10000:10000 -e PORT=10000 credit-card-fraud-api
```

The API will be available at:

http://localhost:10000/

## API Endpoints

### GET /

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy",
  "message": "Credit Card Fraud Detection API is running"
}
```

### POST /predict

Accepts a transaction containing the 30 model input features and returns a prediction.

Example request structure:

```json
{
  "features": {
    "Time": 0,
    "V1": 0,
    "V2": 0,
    "V3": 0,
    "V4": 0,
    "V5": 0,
    "V6": 0,
    "V7": 0,
    "V8": 0,
    "V9": 0,
    "V10": 0,
    "V11": 0,
    "V12": 0,
    "V13": 0,
    "V14": 0,
    "V15": 0,
    "V16": 0,
    "V17": 0,
    "V18": 0,
    "V19": 0,
    "V20": 0,
    "V21": 0,
    "V22": 0,
    "V23": 0,
    "V24": 0,
    "V25": 0,
    "V26": 0,
    "V27": 0,
    "V28": 0,
    "Amount": 100
  }
}
```

Example response:

```json
{
  "prediction": 0,
  "label": "Legitimate",
  "fraud_probability": 0.0000005105
}
```

The response includes the predicted class, its human-readable label, and the model's estimated fraud probability.

## Testing

Run the automated tests with:

```powershell
python -m pytest -v
```

The test suite checks:

- API health endpoint.
- Successful prediction response.
- Validation of missing input features.

## Continuous Integration

GitHub Actions automatically runs the project tests and builds the Docker image when changes are pushed to the configured branches or when pull requests are opened.

Workflow configuration:

`.github/workflows/ci.yml`

## Deployment

The API is deployed as a Docker-based Web Service on Render.

The deployment uses the Dockerfile at `docker/Dockerfile` and automatically deploys changes pushed to the connected GitHub repository.

## Future Improvements

- Add model performance metrics and evaluation visualizations.
- Introduce prediction logging and monitoring.
- Improve input validation and API documentation.
- Add model versioning and retraining workflows.
- Develop a simple user interface for submitting transactions.

## Author

**Rethabile Kgolobe**

GitHub: https://github.com/Rethabile-Kgolobe
