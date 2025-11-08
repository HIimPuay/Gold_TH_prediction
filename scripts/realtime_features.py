# scripts/realtime_features.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

VARS = ["gold","fx","cpi","oil","set"]

def load_context(feature_store_path: Path, context_days: int = 14) -> pd.DataFrame:
    df = pd.read_csv(feature_store_path, parse_dates=["date"])
    df = df.sort_values("date")
    return df.tail(context_days).reset_index(drop=True)

def make_realtime_row(latest_date: pd.Timestamp, payload: Dict[str, float]) -> pd.DataFrame:
    row = {"date": pd.to_datetime(latest_date)}
    for k in VARS:
        row[k] = float(payload[k])
    return pd.DataFrame([row])

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    for col in VARS:
        out[f"{col}_lag1"] = out[col].shift(1)
        out[f"{col}_lag3"] = out[col].shift(3)
        out[f"{col}_roll7_mean"] = out[col].rolling(7, min_periods=3).mean()
        out[f"{col}_pct_change"] = out[col].pct_change()
    return out

def build_feature_vector_for_today(context_df: pd.DataFrame, today_raw_df: pd.DataFrame) -> pd.DataFrame:
    full = pd.concat([context_df[["date"] + VARS], today_raw_df], ignore_index=True)
    full = add_features(full)
    today = full.tail(1).copy()
    features = []
    for col in VARS:
        features += [f"{col}_lag1", f"{col}_lag3", f"{col}_roll7_mean", f"{col}_pct_change"]
    features += VARS  # include raw as well (depends on model)
    X = today[features].dropna().copy()
    return X
