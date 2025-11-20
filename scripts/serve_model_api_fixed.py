#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.pkl"
FEATURE_STORE_PATH = BASE_DIR / "data" / "Feature_store" / "feature_store.csv"

app = FastAPI(title="Gold Price Predictor")

model = None
metadata = None

@app.on_event("startup")
async def load_model():
    global model, metadata
    try:
        model = joblib.load(MODEL_PATH)
        metadata = joblib.load(METADATA_PATH)
        print(f"✅ Model loaded: {metadata.get('model_type')}")
    except Exception as e:
        print(f"❌ Error: {e}")

class PredictInput(BaseModel):
    date: Optional[str] = None
    gold: float
    fx: float
    cpi: float
    oil: float
    set: float
    btc: float = 0

class PredictOutput(BaseModel):
    model: str
    predicted_gold: float
    message: str

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictOutput)
def predict(inp: PredictInput):
    if model is None:
        raise HTTPException(500, "Model not loaded")
    
    try:
        df_context = pd.read_csv(FEATURE_STORE_PATH, parse_dates=['date'])
        df_context = df_context.sort_values('date').tail(14)
        
        new_row = {
            'date': pd.to_datetime(inp.date or datetime.now().strftime("%Y-%m-%d")),
            'gold': inp.gold, 'fx': inp.fx, 'cpi': inp.cpi,
            'oil': inp.oil, 'set': inp.set
        }
        
        df = pd.concat([df_context[['date','gold','fx','cpi','oil','set']], 
                       pd.DataFrame([new_row])], ignore_index=True)
        
        for var in ['gold','fx','cpi','oil','set']:
            df[f'{var}_lag1'] = df[var].shift(1)
            df[f'{var}_lag3'] = df[var].shift(3)
            df[f'{var}_roll7_mean'] = df[var].rolling(7, min_periods=3).mean()
            df[f'{var}_pct_change'] = df[var].pct_change()
        
        X = df.tail(1)[metadata['features']].fillna(0)
        prediction = float(model.predict(X)[0])
        
        return PredictOutput(
            model=metadata.get('model_type', 'unknown'),
            predicted_gold=prediction,
            message=f"Predicted: {prediction:.2f} THB"
        )
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
