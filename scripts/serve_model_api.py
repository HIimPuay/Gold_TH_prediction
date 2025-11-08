# scripts/serve_model_api.py  (BTC-enabled)
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- Import BTC realtime features (อยู่ใน scripts/realtime_features_btc.py) ---
# รองรับหลายกรณี: โครงสร้างแพ็กเกจ/ไม่เป็นแพ็กเกจ
try:
    from scripts.realtime_features_btc import (  # โครงสร้างปัจจุบันของคุณ
        load_context,
        make_realtime_row,
        build_feature_vector_for_today,
        VARS,
    )
except Exception:
    try:
        # เผื่อกรณีรันแบบวางไว้ใน PYTHONPATH โดยตรง
        from realtime_features_btc import (
            load_context,
            make_realtime_row,
            build_feature_vector_for_today,
            VARS,
        )
    except Exception:
        # เผื่อกรณี import แบบ relative (เมื่อรันเป็นโมดูล scripts.serve_model_api)
        from .realtime_features_btc import (
            load_context,
            make_realtime_row,
            build_feature_vector_for_today,
            VARS,
        )

APP_TITLE = "Thai Gold Price Predictor API (BTC-enabled)"

# --- Feature Store path: รองรับได้หลาย layout ---
POSSIBLE_FS = [
    Path("data/Feature_store/feature_store.csv"),
    Path("./feature_store.csv"),
    Path("../data/Feature_store/feature_store.csv"),
]
FEATURE_STORE_PATH = next((p for p in POSSIBLE_FS if p.exists()), POSSIBLE_FS[0])

# --- Model path: โครงสร้างในโปรเจกต์ของคุณคือ 'model/best_model.pkl' ---
POSSIBLE_MODELS = [
    Path("model/best_model.pkl"),   # <-- ของคุณอยู่ตรงนี้
    Path("models/best_model.pkl"),
    Path("./best_model.pkl"),
]
MODEL_PATH = next((p for p in POSSIBLE_MODELS if p.exists()), POSSIBLE_MODELS[0])

CONTEXT_DAYS = 14

app = FastAPI(title=APP_TITLE, version="1.1.1")


class PredictInput(BaseModel):
    date: Optional[str] = Field(default=None, description="ISO date e.g. 2025-11-08")
    gold: float
    fx: float
    cpi: float
    oil: float
    set: float
    btc: float  # <<< added


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
            print(f"[WARN] Failed to load model at {model_path}: {e}")
    return None, True


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_feature_store": FEATURE_STORE_PATH.exists(),
        "feature_store_path": FEATURE_STORE_PATH.as_posix(),
        "model_path": MODEL_PATH.as_posix(),
        "has_model": MODEL_PATH.exists(),
        "vars": VARS,
    }


@app.post("/predict", response_model=PredictOutput)
def predict(inp: PredictInput):
    # ตรวจ Feature Store
    if not FEATURE_STORE_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Feature store not found at {FEATURE_STORE_PATH}",
        )

    # วันที่ใช้คำนวณ
    use_date = inp.date or datetime.utcnow().strftime("%Y-%m-%d")
    today_dt = pd.to_datetime(use_date)

    # โหลด context และทำแถว realtime จาก payload
    context_df = load_context(FEATURE_STORE_PATH, context_days=CONTEXT_DAYS)
    payload = {k: getattr(inp, k) for k in VARS}  # รวม btc ด้วย
    today_raw_df = make_realtime_row(today_dt, payload)

    # สร้างฟีเจอร์สำหรับวันนี้
    X = build_feature_vector_for_today(context_df, today_raw_df)
    if X is None or getattr(X, "empty", False):
        raise HTTPException(
            status_code=400,
            detail="Not enough context to build features (need ≥7 days).",
        )

    # โหลดโมเดล (ถ้าไม่มีจะ fallback baseline)
    model, baseline = load_model(MODEL_PATH)

    if baseline:
        # baseline: ใช้ rolling-7 ของ gold จาก context
        ctx = context_df.copy().sort_values("date")
        ctx["gold_roll7_mean"] = ctx["gold"].rolling(7, min_periods=3).mean()
        last_roll = float(ctx["gold_roll7_mean"].dropna().tail(1).values[0])
        return PredictOutput(
            model="baseline_roll7",
            predicted_gold=last_roll,
            used_baseline=True,
            message="best_model.pkl not found; used rolling-7 mean as baseline",
        )

    # พยากรณ์จากโมเดล
    try:
        # รองรับทั้ง DataFrame/ndarray
        input_array = X.values if hasattr(X, "values") else X
        y_pred = model.predict(input_array)
        pred = float(y_pred[0])
        return PredictOutput(
            model=type(model).__name__,
            predicted_gold=pred,
            used_baseline=False,
            message="ok",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")


@app.get("/version")
def version():
    return {"app": APP_TITLE, "version": "1.1.1", "vars": VARS}
