#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_feature_store.py
Quick QA for feature_store.csv
"""
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from textwrap import indent

REQUIRED_COLS = [
    "date","gold","fx","cpi","oil","set","gold_next",
    "gold_lag1","gold_lag3","fx_lag1","fx_lag3","cpi_lag1","cpi_lag3","oil_lag1","oil_lag3","set_lag1","set_lag3",
    "gold_roll7_mean","fx_roll7_mean","cpi_roll7_mean","oil_roll7_mean","set_roll7_mean",
    "gold_pct_change","fx_pct_change","cpi_pct_change","oil_pct_change","set_pct_change"
]
NUMERIC_COLS = [c for c in REQUIRED_COLS if c not in ["date"]]

def parse_args():
    ap = argparse.ArgumentParser(description="Validate feature_store.csv")
    ap.add_argument("--path", type=Path, default=Path("../data/Feature_store/feature_store.csv"))
    ap.add_argument("--strict", action="store_true")
    return ap.parse_args()

def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr); return 1

def warn(msg):
    print(f"[WARN] {msg}")

def ok(msg):
    print(f"[OK] {msg}")

def main():
    args = parse_args()
    p = args.path
    if not p.exists():
        return fail(f"File not found: {p}")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return fail(f"Cannot read CSV: {e}")
    if "date" not in df.columns:
        return fail("Missing 'date'")
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except Exception as e:
        return fail(f"date parse failed: {e}")
    if not df["date"].is_monotonic_increasing: return fail("date not monotonic")
    if not df["date"].is_unique: return fail("date duplicates")

    ok(f"Date range: {df['date'].min().date()} → {df['date'].max().date()} (rows={len(df)})")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: return fail(f"Missing required columns: {missing}")

    non_numeric = [c for c in NUMERIC_COLS if not np.issubdtype(df[c].dtype, np.number)]
    if non_numeric: return fail(f"Non-numeric columns: {non_numeric}")

    na_counts = df[REQUIRED_COLS].isna().sum()
    if int(na_counts.sum()) > 0:
        print(na_counts[na_counts>0])
        return fail("NA present in required columns")
    ok("No NaN values in required columns")

    errors = 0; warnings = 0
    for col in ["gold","cpi","oil","set"]:
        if (df[col] <= 0).any():
            errors += 1; print(df.loc[df[col] <= 0, ["date", col]].head())
            print(f"[FAIL] {col} contains non-positive values")

    if (df["fx"] < 20).any() or (df["fx"] > 50).any():
        warnings += 1; warn("fx outside 20..50 — verify source/units")

    for col in ["gold","fx","cpi","oil","set"]:
        pct = df[f"{col}_pct_change"].abs()
        if (pct > 0.2).any():
            cnt = int((pct > 0.2).sum()); warnings += 1
            warn(f"{col}_pct_change has {cnt} spikes > 20% in a day — check anomalies")

    keep = ["gold","gold_next","fx","cpi","oil","set"]
    stats = df[keep].describe().T[["mean","std","min","max"]]
    print("\nQuick stats:\n" + indent(stats.to_string(), "  "))

    if errors > 0: return 1
    if args.strict and warnings > 0: return 1
    ok("Validation passed"); return 0

if __name__ == "__main__":
    sys.exit(main())
