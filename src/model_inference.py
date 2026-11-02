import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REQUIRED = ["area", "bedrooms", "bathrooms", "age", "location", "property_type"]

def load_model():
    current_path = MODEL_DIR / "current.json"
    if not current_path.exists():
        raise FileNotFoundError("No trained model found. Run: python -m src.model_training")
    metadata = json.loads(current_path.read_text())
    artifact = MODEL_DIR / metadata["artifact"]
    if not artifact.exists():
        raise FileNotFoundError(f"Model artifact missing: {artifact}")
    return joblib.load(artifact), metadata

def validate_input(payload):
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object.")
    missing = [key for key in REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    row = {}
    for key in ["area", "bedrooms", "bathrooms", "age"]:
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number.")
        if value < 0:
            raise ValueError(f"{key} cannot be negative.")
        row[key] = value
    if row["area"] <= 0:
        raise ValueError("area must be greater than zero.")
    for key in ["location", "property_type"]:
        value = payload[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string.")
        row[key] = value.strip()
    return pd.DataFrame([row])

def predict(payload):
    frame = validate_input(payload)
    model, metadata = load_model()
    prediction = float(model.predict(frame)[0])
    return {"predicted_price_inr": round(max(0.0, prediction), 2), "model_version": metadata["version"], "algorithm": metadata["algorithm"]}
