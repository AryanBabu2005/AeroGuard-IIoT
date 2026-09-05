import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
# ... rest of your imports ...
import joblib
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# -------------------------------------------------------------
# Test Group 1: Model & Pipeline Artifact Checks
# -------------------------------------------------------------
def test_artifacts_exist():
    """Verify that all serialized pipeline models exist on disk."""
    assert os.path.exists("models/rf_baseline.pkl"), "Random forest model artifact missing"
    assert os.path.exists("models/scaler.pkl"), "Scaler artifact missing"

def test_scaler_transform_dimension():
    """Verify StandardScaler expects and transforms 15 raw sensor features."""
    scaler = joblib.load("models/scaler.pkl")
    assert getattr(scaler, "n_features_in_", 15) == 15, "Scaler must expect 15 features"
    
    mock_raw_15 = np.random.randn(5, 15)
    scaled = scaler.transform(mock_raw_15)
    assert scaled.shape == (5, 15), "Scaler output dimension must be (N, 15)"

def test_model_inference_dimensions():
    """Verify Random Forest accepts the expanded 30-feature vector."""
    model = joblib.load("models/rf_baseline.pkl")
    assert getattr(model, "n_features_in_", 30) == 30, "Random Forest must expect 30 features"
    
    # 15 scaled sensor means + 15 sensor standard deviations
    mock_expanded_30 = np.random.randn(1, 30)
    pred = model.predict(mock_expanded_30)
    assert len(pred) == 1
    assert not np.isnan(pred[0])
    assert isinstance(float(pred[0]), float)

# -------------------------------------------------------------
# Test Group 2: End-to-End FastAPI Endpoint Verification
# -------------------------------------------------------------
def test_api_health():
    """Verify the /health endpoint reports ready and correct feature counts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["scaler_features_in"] == 15
    assert data["model_features_in"] == 30

def test_api_prediction_valid():
    """Verify full end-to-end inference via the REST endpoint with 15 sensors."""
    payload = {
        "engine_id": 1,
        "cycle": 150,
        "features": [
            642.35, 1589.70, 1404.66, 553.75, 2388.05,
            9052.12, 47.35, 521.66, 2388.08, 8138.62,
            8.4195, 392.0, 39.06, 23.4190, 100.0
        ]
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predicted_rul" in body
    assert body["predicted_rul"] >= 0.0
    assert body["status"] in ["OPTIMAL", "WARNING", "CRITICAL"]
    assert "confidence_std" in body

def test_api_prediction_invalid_dimension():
    """Verify the endpoint returns HTTP 422 if the wrong number of features is passed."""
    payload = {
        "engine_id": 1,
        "cycle": 150,
        "features": [1.0, 2.0, 3.0] # Only 3 features instead of 15
    }
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422