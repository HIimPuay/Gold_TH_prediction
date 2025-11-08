#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_alignment_steps.py (with Bitcoin support)
---------------------------------
Detailed step-by-step script for:
(2) Data Alignment
(3) Frequency Transformation
(4) Handle Missing Values
(5) Target Variable
(6) Feature Engineering (Lag / Rolling / pct_change)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------- CONFIG ---------------- #
RAW = Path("../data/raw")              # path to raw data
OUT = Path("../data/Feature_store/feature_store.csv")
BTC_PATH = RAW / "bitcoin_history.csv"  # <<<< เพิ่ม Bitcoin

# ---------------- STEP 2: Data Alignment ---------------- #

def parse_buddhist_date(s):
    """แปลงวันที่ พ.ศ. เช่น 02/01/2566 → 2023-01-02"""
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if isinstance(s, str) and "/" in s:
        try:
            d, m, y = s.split("/")[:3]
            y = int(y) - 543 if int(y) > 2400 else int(y)
            dt = pd.to_datetime(f"{d}/{m}/{y}", dayfirst=True, errors="coerce")
        except Exception:
            pass
    if pd.notna(dt) and dt.year > 2400:
        dt = dt.replace(year=dt.year - 543)
    return dt


print("=== STEP 2: Data Alignment ===")

# Gold
gold = pd.read_csv(RAW / "gold_history.csv")
gold["date"] = gold["date"].apply(parse_buddhist_date)
gold = gold.dropna(subset=["date"])
price_col = "gold_sell" if "gold_sell" in gold.columns else "gold_bar_sell"
gold = gold[["date", price_col]].rename(columns={price_col: "gold"}).sort_values("date")

# FX (monthly)
fx = pd.read_csv(RAW / "exchange_rate.csv")
fx["date"] = pd.to_datetime(fx["period"].astype(str) + "-01", errors="coerce")
fx = fx[["date", "mid_rate"]].rename(columns={"mid_rate": "fx"}).sort_values("date")

# CPI (monthly, code == 0)
cpi = pd.read_csv(RAW / "CPI_clean_for_supabase.csv")
cpi = cpi[cpi["code"].astype(str) == "0"].copy()
cpi["date"] = pd.to_datetime(cpi["date"], errors="coerce")
cpi = cpi[["date", "cpi_index"]].rename(columns={"cpi_index": "cpi"}).sort_values("date")

# OIL (monthly, mean per month)
oil = pd.read_csv(RAW / "petroleum_data.csv")
oil["date"] = pd.to_datetime(oil["period"].astype(str) + "-01", errors="coerce")
oil = oil.groupby("date", as_index=False)["value"].mean().rename(columns={"value": "oil"})

# SET (daily)
setidx = pd.read_csv(RAW / "set_index.csv")
setidx["date_parsed"] = pd.to_datetime(setidx["date"], errors="coerce")
setidx = setidx[setidx["date_parsed"].notna()].copy()
setidx["date"] = setidx["date_parsed"]
setidx["Close"] = pd.to_numeric(setidx["Close"], errors="coerce")
setidx = setidx.dropna(subset=["Close"])
setidx = setidx[["date", "Close"]].rename(columns={"Close": "set"}).sort_values("date")

# <<<< เพิ่ม Bitcoin >>>>
btc = None
if BTC_PATH.exists():
    try:
        btc_df = pd.read_csv(BTC_PATH)
        date_col = "Date" if "Date" in btc_df.columns else "date"
        price_col = "Close" if "Close" in btc_df.columns else "close"
        btc = btc_df[[date_col, price_col]].copy()
        btc["date"] = pd.to_datetime(btc[date_col], errors="coerce")
        btc = btc.rename(columns={price_col: "btc"})
        btc = btc[["date", "btc"]].dropna().sort_values("date")
        btc["btc"] = pd.to_numeric(btc["btc"], errors="coerce")
        btc = btc.dropna(subset=["btc"]).drop_duplicates(subset=["date"], keep="last")
        print(f"✅ Bitcoin data loaded: {len(btc)} rows")
    except Exception as e:
        print(f"⚠️ Bitcoin load failed: {e}")
        btc = None
else:
    print(f"⚠️ Bitcoin file not found at {BTC_PATH}")

# Combine by date
datasets = [gold, fx, cpi, oil, setidx]
if btc is not None:
    datasets.append(btc)

start = min(df["date"].min() for df in datasets)
end = max(df["date"].max() for df in datasets)
calendar = pd.DataFrame({"date": pd.bdate_range(start, end)})

feat = calendar.copy()
for name, df in [("gold", gold), ("fx", fx), ("cpi", cpi), ("oil", oil), ("set", setidx)]:
    feat = feat.merge(df, on="date", how="left")

if btc is not None:
    feat = feat.merge(btc, on="date", how="left")

print("Combined shape:", feat.shape)

# ---------------- STEP 3: Frequency Transformation ---------------- #
print("=== STEP 3: Frequency Transformation ===")

feat["cpi"] = feat["cpi"].ffill()
feat["fx"] = feat["fx"].ffill()
feat["oil"] = feat["oil"].ffill()

cols_to_fill = ["gold", "set"]
if "btc" in feat.columns:
    cols_to_fill.append("btc")

for col in cols_to_fill:
    feat[col] = feat[col].ffill().bfill()

# ---------------- STEP 4: Handle Missing Values ---------------- #
print("=== STEP 4: Handle Missing Values ===")
cols_to_interpolate = ["gold", "fx", "cpi", "oil", "set"]
if "btc" in feat.columns:
    cols_to_interpolate.append("btc")

for col in cols_to_interpolate:
    feat[col] = feat[col].ffill().bfill().interpolate(limit_direction="both")

# ---------------- STEP 5: Target Variable ---------------- #
print("=== STEP 5: Target Variable ===")
feat["gold_next"] = feat["gold"].shift(-1)

# ---------------- STEP 6: Feature Engineering ---------------- #
print("=== STEP 6: Feature Engineering ===")
vars_all = ["gold", "fx", "cpi", "oil", "set"]
if "btc" in feat.columns:
    vars_all.append("btc")

for col in vars_all:
    feat[f"{col}_lag1"] = feat[col].shift(1)
    feat[f"{col}_lag3"] = feat[col].shift(3)
    feat[f"{col}_roll7_mean"] = feat[col].rolling(7, min_periods=3).mean()
    feat[f"{col}_pct_change"] = feat[col].pct_change()

feat = feat.dropna().reset_index(drop=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
feat.to_csv(OUT, index=False)

print(f"[OK] Saved Feature Store → {OUT.as_posix()}")
print("Rows:", len(feat), "Cols:", len(feat.columns))
print("Variables:", vars_all)