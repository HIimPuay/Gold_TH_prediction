#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

# ใช้ absolute path
BASE_DIR = Path("/Users/nichanun/Desktop/DSDN")
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.pkl"
FEATURE_STORE = BASE_DIR / "data" / "Feature_store" / "feature_store.csv"

app = FastAPI(title="Gold Price API")

# Global variables
model = None
metadata = None

@app.on_event("startup")
def startup():
    global model, metadata
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        print(f"✅ Model loaded: {metadata.get('model_type')}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

class Input(BaseModel):
    date: Optional[str] = None
    gold: float
    fx: float
    cpi: float
    oil: float
    set: float
    btc: float = 0

@app.get("/")
def root():
    return {"status": "running", "model": metadata.get('model_type') if metadata else None}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_type": metadata.get('model_type') if metadata else None
    }

@app.post("/predict")
def predict(inp: Input):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        # โหลด context
        df = pd.read_csv(FEATURE_STORE, parse_dates=['date'])
        df = df.sort_values('date').tail(14)
        
        # เพิ่มแถวใหม่
        from datetime import datetime
        new = pd.DataFrame([{
            'date': pd.to_datetime(inp.date or datetime.now().strftime("%Y-%m-%d")),
            'gold': inp.gold, 'fx': inp.fx, 'cpi': inp.cpi,
            'oil': inp.oil, 'set': inp.set
        }])
        
        df = pd.concat([df[['date','gold','fx','cpi','oil','set']], new], ignore_index=True)
        
        # สร้าง features
        for v in ['gold','fx','cpi','oil','set']:
            df[f'{v}_lag1'] = df[v].shift(1)
            df[f'{v}_lag3'] = df[v].shift(3)
            df[f'{v}_roll7_mean'] = df[v].rolling(7, min_periods=3).mean()
            df[f'{v}_pct_change'] = df[v].pct_change()
        
        # Predict
        X = df.tail(1)[metadata['features']].fillna(0)
        pred = float(model.predict(X)[0])
        
        return {
            "model": metadata.get('model_type'),
            "predicted_gold": pred,
            "message": f"Success: {pred:.2f} THB"
        }
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    print(f"Base: {BASE_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"Exists: {MODEL_PATH.exists()}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
