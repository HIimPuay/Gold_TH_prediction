#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate feature_store.csv (with optional btc columns)
"""
import sys, argparse
import pandas as pd
import numpy as np
from pathlib import Path

BASE_REQ = [
    "date","gold","fx","cpi","oil","set","gold_next",
    "gold_lag1","gold_lag3","fx_lag1","fx_lag3","cpi_lag1","cpi_lag3","oil_lag1","oil_lag3","set_lag1","set_lag3",
    "gold_roll7_mean","fx_roll7_mean","cpi_roll7_mean","oil_roll7_mean","set_roll7_mean",
    "gold_pct_change","fx_pct_change","cpi_pct_change","oil_pct_change","set_pct_change"
]
BTC_REQ = [
    "btc","btc_lag1","btc_lag3","btc_roll7_mean","btc_pct_change"
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, default=Path("../data/Feature_store/feature_store.csv"))
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()

def fail(msg): print(f"[FAIL] {msg}", file=sys.stderr); return 1
def ok(msg): print(f"[OK] {msg}")

def main():
    a = parse_args()
    p = a.path
    if not p.exists(): return fail(f"File not found: {p}")
    df = pd.read_csv(p)
    if "date" not in df.columns: return fail("Missing 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    if not df["date"].is_monotonic_increasing: return fail("date not monotonic")
    if not df["date"].is_unique: return fail("date duplicates")
    ok(f"Date range {df['date'].min().date()} → {df['date'].max().date()} rows={len(df)}")

    required = BASE_REQ.copy()
    if "btc" in df.columns:
        required += BTC_REQ

    missing = [c for c in required if c not in df.columns]
    if missing: return fail(f"Missing columns: {missing}")

    nonnum = [c for c in required if c!="date" and not np.issubdtype(df[c].dtype, np.number)]
    if nonnum: return fail(f"Non-numeric detected: {nonnum}")

    if df[required].isna().sum().sum() > 0: return fail("NA present in required columns")

    ok("Validation passed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
