#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "model_metadata.pkl"
FEATURE_STORE_PATH = BASE_DIR / "data" / "Feature_store" / "feature_store.csv"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Gold Price Predictor API",
    description="API สำหรับทำนายราคาทองแท่ง (รับซื้อ)",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model = None
metadata = None

@app.on_event("startup")
async def load_model():
    global model, metadata
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded: {MODEL_PATH}")
        if METADATA_PATH.exists():
            metadata = joblib.load(METADATA_PATH)
            print(f"✅ Metadata loaded: {metadata.get('model_type', 'unknown').upper()}")
    except Exception as e:
        print(f"❌ Error: {e}")

class PredictInput(BaseModel):
    date: Optional[str] = None
    gold: float
    fx: float
    cpi: float
    oil: float
    set: float
    btc: Optional[float] = 0.0  # ทำให้เป็น Optional

class PredictOutput(BaseModel):
    model: str
    predicted_gold: float
    change: float
    change_pct: float
    message: str

@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path) as f:
            return HTMLResponse(content=f.read())
    return {"message": "Gold Price Predictor API", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "metadata_loaded": metadata is not None,
        "feature_store_exists": FEATURE_STORE_PATH.exists()
    }

@app.post("/predict", response_model=PredictOutput)
def predict(inp: PredictInput):
    if model is None or metadata is None:
        raise HTTPException(500, "Model not loaded")
    
    if not FEATURE_STORE_PATH.exists():
        raise HTTPException(500, "Feature store not found")
    
    try:
        # โหลด context
        df_context = pd.read_csv(FEATURE_STORE_PATH, parse_dates=['date'])
        df_context = df_context.sort_values('date').tail(14)
        
        # สร้างแถวใหม่
        new_date = pd.to_datetime(inp.date if inp.date else datetime.now().strftime("%Y-%m-%d"))
        new_row = {
            'date': new_date,
            'gold': inp.gold,
            'fx': inp.fx,
            'cpi': inp.cpi,
            'oil': inp.oil,
            'set': inp.set
        }
        
        # รวมข้อมูล
        df = pd.concat([
            df_context[['date', 'gold', 'fx', 'cpi', 'oil', 'set']], 
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        # สร้าง features
        for var in ['gold', 'fx', 'cpi', 'oil', 'set']:
            df[f'{var}_lag1'] = df[var].shift(1)
            df[f'{var}_lag3'] = df[var].shift(3)
            df[f'{var}_roll7_mean'] = df[var].rolling(7, min_periods=3).mean()
            df[f'{var}_pct_change'] = df[var].pct_change()
        
        # เตรียม input
        X = df.tail(1)[metadata['features']].fillna(0)
        
        # ทำนาย
        prediction = float(model.predict(X)[0])
        change = prediction - inp.gold
        change_pct = (change / inp.gold * 100) if inp.gold > 0 else 0
        
        return PredictOutput(
            model=metadata.get('model_type', 'unknown').upper(),
            predicted_gold=round(prediction, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            message="ทำนายสำเร็จ"
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Error: {str(e)}")

@app.get("/version")
def version():
    return {
        "api_version": "1.0.0",
        "model_type": metadata.get('model_type', 'unknown') if metadata else None
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Gold Price Predictor API")
    print(f"🌐 Web UI: http://localhost:{port}")
    print(f"📖 API Docs: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
