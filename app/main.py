from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Load trained XGBoost model
MODEL_PATH = BASE_DIR / "artifacts" / "model.pkl"
model = joblib.load(MODEL_PATH)

# Features used during model training
FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions.",
    version="1.0.0"
)


class Transaction(BaseModel):
    features: dict[str, float]


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "Credit Card Fraud Detection API is running"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    supplied_features = set(transaction.features.keys())
    required_features = set(FEATURES)

    missing_features = required_features - supplied_features
    unexpected_features = supplied_features - required_features

    if missing_features or unexpected_features:
        raise HTTPException(
            status_code=400,
            detail={
                "missing_features": sorted(missing_features),
                "unexpected_features": sorted(unexpected_features)
            }
        )

    # Preserve the exact feature order used during training
    input_data = pd.DataFrame(
        [[transaction.features[feature] for feature in FEATURES]],
        columns=FEATURES
    )

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": probability
    }
