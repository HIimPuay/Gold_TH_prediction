#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Feature Store (with Bitcoin as additional factor)
Adds 'btc' features alongside gold, fx, cpi, oil, set.
- Align to daily business calendar
- Handle missing (ffill/bfill/interpolate)
- Create target gold_next, lags (1,3), rolling mean (7d), pct_change
Usage:
  python build_feature_store_btc.py \
    --data-dir ./data/raw \
    --btc ./data/raw/bitcoin_history.csv \
    --out ./data/Feature_store/feature_store.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--btc", type=Path, default=None, help="Path to bitcoin_history.csv")
    ap.add_argument("--out", type=Path, default=Path("feature_store.csv"))
    ap.add_argument("--min-periods", type=int, default=3)
    ap.add_argument("--roll-window", type=int, default=7)
    return ap.parse_args()

def parse_buddhist_date(s):
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if isinstance(s, str) and "/" in s:
        try:
            d,m,y = s.split("/")[:3]
            y = int(y) - 543 if int(y) > 2400 else int(y)
            dt = pd.to_datetime(f"{d}/{m}/{y}", dayfirst=True, errors="coerce")
        except Exception:
            pass
    if pd.notna(dt) and dt.year > 2400:
        dt = dt.replace(year=dt.year - 543)
    return dt

def load_gold(p):
    df = pd.read_csv(p)
    df["date"] = df["date"].apply(parse_buddhist_date)
    df = df.dropna(subset=["date"])
    val_col = "gold_sell" if "gold_sell" in df.columns else ("gold_bar_sell" if "gold_bar_sell" in df.columns else None)
    if val_col is None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols:
            raise ValueError("gold_history.csv: no numeric price column")
        val_col = num_cols[-1]
    out = df[["date", val_col]].rename(columns={val_col:"gold"}).sort_values("date")
    out["gold"] = pd.to_numeric(out["gold"], errors="coerce")
    out = out.dropna(subset=["gold"]).drop_duplicates(subset=["date"], keep="last")
    return out

def load_fx(p):
    df = pd.read_csv(p)
    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"].astype(str) + "-01", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("exchange_rate.csv: need 'period' or 'date'")
    rate_col = "mid_rate" if "mid_rate" in df.columns else ("selling" if "selling" in df.columns else "buying_transfer")
    out = df.dropna(subset=["date"])[["date", rate_col]].rename(columns={rate_col:"fx"}).sort_values("date")
    out["fx"] = pd.to_numeric(out["fx"], errors="coerce")
    out = out.dropna(subset=["fx"]).drop_duplicates(subset=["date"], keep="last")
    return out

def load_cpi(p):
    df = pd.read_csv(p)
    if "code" in df.columns:
        df = df[df["code"].astype(str)=="0"].copy()
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    val_col = "cpi_index" if "cpi_index" in df.columns else "value"
    out = df.dropna(subset=["date"])[["date", val_col]].rename(columns={val_col:"cpi"}).sort_values("date")
    out["cpi"] = pd.to_numeric(out["cpi"], errors="coerce")
    out = out.dropna(subset=["cpi"]).drop_duplicates(subset=["date"], keep="last")
    return out

def load_oil(p):
    df = pd.read_csv(p)
    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"].astype(str) + "-01", errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        raise ValueError("petroleum_data.csv: need 'period' or 'date'")
    val_col = "value" if "value" in df.columns else df.select_dtypes(include=np.number).columns[-1]
    tmp = df.dropna(subset=["date"])[["date", val_col]].rename(columns={val_col:"oil"})
    out = tmp.groupby("date", as_index=False)["oil"].mean().sort_values("date")
    out["oil"] = pd.to_numeric(out["oil"], errors="coerce")
    out = out.dropna(subset=["oil"])
    return out

def load_set(p):
    df = pd.read_csv(p)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_parsed"].notna()].copy()
    df["date"] = df["date_parsed"]
    close_col = "Close" if "Close" in df.columns else df.select_dtypes(include=np.number).columns[0]
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    out = df.dropna(subset=[close_col])[["date", close_col]].rename(columns={close_col:"set"}).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

def load_btc(p):
    df = pd.read_csv(p)
    # Expect columns: Date, Open, High, Low, Close, Volume
    date_col = "Date" if "Date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    price_col = "Close" if "Close" in df.columns else "close"
    out = df.dropna(subset=["date", price_col])[["date", price_col]].rename(columns={price_col:"btc"}).sort_values("date")
    out["btc"] = pd.to_numeric(out["btc"], errors="coerce")
    out = out.dropna(subset=["btc"]).drop_duplicates(subset=["date"], keep="last")
    return out

def main():
    args = parse_args()
    D = args.data_dir
    gold = load_gold(D/"gold_history.csv")
    fx   = load_fx(D/"exchange_rate.csv")
    cpi  = load_cpi(D/"CPI_clean_for_supabase.csv")
    oil  = load_oil(D/"petroleum_data.csv")
    seti = load_set(D/"set_index.csv")
    btc  = load_btc(args.btc) if args.btc else None

    start = min(df["date"].min() for df in [gold, fx, cpi, oil, seti] + ([btc] if btc is not None else []))
    end   = max(df["date"].max() for df in [gold, fx, cpi, oil, seti] + ([btc] if btc is not None else []))
    calendar = pd.DataFrame({"date": pd.bdate_range(start, end)})

    feat = calendar.copy()
    for name, df_ in [("gold",gold),("fx",fx),("cpi",cpi),("oil",oil),("set",seti)]:
        feat = feat.merge(df_, on="date", how="left")
    if btc is not None:
        feat = feat.merge(btc, on="date", how="left")

    # frequency alignment
    feat["cpi"] = feat["cpi"].ffill()
    for col in ["gold","fx","oil","set","btc"] if "btc" in feat.columns else ["gold","fx","oil","set"]:
        feat[col] = feat[col].ffill().bfill()

    for col in [c for c in ["gold","fx","cpi","oil","set","btc"] if c in feat.columns]:
        if feat[col].isna().any():
            feat[col] = feat[col].interpolate(limit_direction="both")

    # target
    feat["gold_next"] = feat["gold"].shift(-1)

    vars_all = [c for c in ["gold","fx","cpi","oil","set","btc"] if c in feat.columns]
    for col in vars_all:
        feat[f"{col}_lag1"] = feat[col].shift(1)
        feat[f"{col}_lag3"] = feat[col].shift(3)
        feat[f"{col}_roll{args.roll_window}_mean"] = feat[col].rolling(args.roll_window, min_periods=args.min_periods).mean()
        feat[f"{col}_pct_change"] = feat[col].pct_change()

    feat = feat.dropna().reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.out, index=False)
    print("[OK] Feature store saved:", args.out.as_posix(), "rows:", len(feat), "cols:", len(feat.columns))

if __name__ == "__main__":
    main()
