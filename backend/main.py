from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="Flight Delay Prediction API",
    description="ML-powered flight delay prediction using Random Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "flight_delay_model.pkl"
model = None
feature_columns = None

@app.on_event("startup")
def load_model():
    global model, feature_columns
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        model = data["model"]
        feature_columns = data["feature_columns"]
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print("⚠️  No model file found. Run train_model.py first!")

class FlightInput(BaseModel):
    airline: str
    month: int
    origin: str
    destination: str
    day_of_week: int
    departure_hour: int
    distance: int

AIRLINE_CODES = ["AA", "AS", "B6", "DL", "F9", "NK", "UA", "WN"]
AIRPORT_CODES = ["ATL", "DEN", "DFW", "JFK", "LAS", "LAX", "MIA", "ORD", "SEA", "SFO"]

def encode_features(data: FlightInput) -> pd.DataFrame:
    row = {}
    row["MONTH"]        = data.month
    row["DAY_OF_WEEK"]  = data.day_of_week
    row["DEP_HOUR"]     = data.departure_hour
    row["DISTANCE"]     = data.distance
    row["IS_WEEKEND"]   = 1 if data.day_of_week in [6, 7] else 0
    row["IS_PEAK_HOUR"] = 1 if data.departure_hour in range(16, 21) else 0
    row["IS_WINTER"]    = 1 if data.month in [12, 1, 2] else 0
    row["IS_SUMMER"]    = 1 if data.month in [6, 7, 8] else 0
    for code in AIRLINE_CODES:
        row[f"AIRLINE_{code}"] = 1 if data.airline == code else 0
    for code in AIRPORT_CODES:
        row[f"ORIGIN_{code}"] = 1 if data.origin == code else 0
        row[f"DEST_{code}"]   = 1 if data.destination == code else 0
    df = pd.DataFrame([row])
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
    return df

FEATURE_IMPORTANCE = [
    {"feature": "Airline",   "importance": 0.28},
    {"feature": "Dep. Hour", "importance": 0.22},
    {"feature": "Month",     "importance": 0.18},
    {"feature": "Origin",    "importance": 0.16},
    {"feature": "Day",       "importance": 0.10},
    {"feature": "Distance",  "importance": 0.06},
]

def get_risk_level(proba: float) -> str:
    if proba >= 0.70:
        return "High"
    elif proba >= 0.45:
        return "Medium"
    else:
        return "Low"

@app.post("/predict")
def predict(flight: FlightInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    try:
        features = encode_features(flight)
        proba = model.predict_proba(features)[0][1]
        delayed = bool(proba >= 0.5)
        estimated_delay = 0
        if delayed:
            estimated_delay = int(proba * 90 + 10)
        return {
            "delayed": delayed,
            "confidence": round(float(proba) if delayed else float(1 - proba), 2),
            "delay_probability": round(float(proba), 2),
            "estimated_delay_minutes": estimated_delay,
            "risk_level": get_risk_level(proba),
            "feature_importance": FEATURE_IMPORTANCE,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}