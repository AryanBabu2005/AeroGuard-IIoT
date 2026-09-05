import traceback
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="AeroGuard IIoT - Telemetry Engine API",
    description="Real-time RUL prediction and stochastic uncertainty quantification microservice",
    version="1.0.0"
)

# 1. Load trained artifacts
try:
    model = joblib.load("models/rf_baseline.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    model = None
    scaler = None

class TelemetryPayload(BaseModel):
    engine_id: int = Field(..., example=1)
    cycle: int = Field(..., example=150)
    features: List[float] = Field(
        ..., 
        description="15 raw selected sensor telemetry values"
    )

class PredictionResponse(BaseModel):
    engine_id: int
    cycle: int
    predicted_rul: float
    status: str
    confidence_std: float

@app.get("/")
def root():
    return {
        "service": "AeroGuard IIoT - Telemetry Engine API",
        "docs": "/docs",
        "health": "/health",
        "status": "online"
    }

@app.get("/health")
def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Pipeline artifacts missing.")
    
    return {
        "status": "ready",
        "scaler_features_in": getattr(scaler, "n_features_in_", 15),
        "model_features_in": getattr(model, "n_features_in_", 30)
    }

@app.post("/v1/predict", response_model=PredictionResponse)
def predict_telemetry(payload: TelemetryPayload):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded")
    
    try:
        raw_vals = np.array(payload.features, dtype=float)
        
        # Step 1: Validate the 15 input sensors
        if raw_vals.shape[0] != 15:
            raise HTTPException(
                status_code=422,
                detail=f"Dimension mismatch: Scaler requires exactly 15 sensor features, received {raw_vals.shape[0]}."
            )

        # Step 2: Scale the 15 raw sensors (Scaler expects 15)
        scaled_15 = scaler.transform(raw_vals.reshape(1, -1))

        # Step 3: Expand to 30 features (15 scaled values + 15 standard deviations / jitter proxies)
        # DecisionTreeRegressor expects 30 features
        expanded_30 = np.hstack([scaled_15, np.zeros((1, 15))])

        # Step 4: Tree-level predictions for ensemble uncertainty
        tree_preds = [float(tree.predict(expanded_30)[0]) for tree in model.estimators_]
        mean_rul = float(np.mean(tree_preds))
        std_rul = float(np.std(tree_preds))
        
        # Operational classification
        if mean_rul < 30:
            status = "CRITICAL"
        elif mean_rul < 75:
            status = "WARNING"
        else:
            status = "OPTIMAL"

        return PredictionResponse(
            engine_id=payload.engine_id,
            cycle=payload.cycle,
            predicted_rul=round(max(0.0, mean_rul), 2),
            status=status,
            confidence_std=round(std_rul, 2)
        )

    except HTTPException:
        raise
    except Exception as err:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(err))