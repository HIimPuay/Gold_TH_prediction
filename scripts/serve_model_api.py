# scripts/serve_model_api.py
from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd

from realtime_features import load_context, make_realtime_row, build_feature_vector_for_today, VARS

APP_TITLE = "Thai Gold Price Predictor API"
FEATURE_STORE_PATH = Path("data/Feature_store/feature_store.csv")
MODEL_PATH = Path("models/best_model.pkl")
CONTEXT_DAYS = 14

app = FastAPI(title=APP_TITLE, version="1.0.0")

class PredictInput(BaseModel):
    date: Optional[str] = Field(default=None, description="ISO date e.g. 2025-11-07")
    gold: float
    fx: float
    cpi: float
    oil: float
    set: float

class PredictOutput(BaseModel):
    model: str
    predicted_gold: float
    used_baseline: bool
    message: str

def load_model(model_path: Path):
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            return model, False
        except Exception as e:
            print(f"[WARN] Failed to load model: {e}")
    return None, True

@app.get("/health")
def health():
    return {"status": "ok", "has_feature_store": FEATURE_STORE_PATH.exists()}

@app.post("/predict", response_model=PredictOutput)
def predict(inp: PredictInput):
    if not FEATURE_STORE_PATH.exists():
        raise HTTPException(status_code=500, detail="Feature store not found")

    use_date = inp.date or datetime.utcnow().strftime("%Y-%m-%d")
    today_dt = pd.to_datetime(use_date)

    context_df = load_context(FEATURE_STORE_PATH, context_days=CONTEXT_DAYS)
    payload = {k: getattr(inp, k) for k in VARS}
    today_raw_df = make_realtime_row(today_dt, payload)

    X = build_feature_vector_for_today(context_df, today_raw_df)
    if X.empty:
        raise HTTPException(status_code=400, detail="Not enough context to build features (need ≥7 days).")

    model, baseline = load_model(MODEL_PATH)

    if baseline:
        ctx = context_df.copy().sort_values("date")
        ctx["gold_roll7_mean"] = ctx["gold"].rolling(7, min_periods=3).mean()
        last_roll = float(ctx["gold_roll7_mean"].dropna().tail(1).values[0])
        return PredictOutput(model="baseline_roll7", predicted_gold=last_roll, used_baseline=True,
                             message="best_model.pkl not found; used rolling-7 mean as baseline")

    try:
        y_pred = model.predict(X.values)
        pred = float(y_pred[0])
        return PredictOutput(model=type(model).__name__, predicted_gold=pred, used_baseline=False, message="ok")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

@app.get("/version")
def version():
    return {"app": APP_TITLE, "version": "1.0.0"}
