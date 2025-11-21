
#!/usr/bin/env python3
"""
build_feature_store_fixed.py - สร้าง Feature Store (แก้ชื่อ column)

แก้ไข:
- ใช้ _roll7_mean แทน _roll7
- ใช้ _pct_change แทน _pct
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "Feature_store" / "feature_store.csv"

THAI_MONTHS = {
    "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", "เมษายน": "04",
    "พฤษภาคม": "05", "มิถุนายน": "06", "กรกฎาคม": "07", "สิงหาคม": "08",
    "กันยายน": "09", "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12"
}

def parse_buddhist_date(s):
    """แปลง พ.ศ. เป็น ค.ศ."""
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if isinstance(s, str) and "/" in s:
        try:
            d, m, y = s.split("/")[:3]
            y = int(y)
            if y > 2400:
                y -= 543
            dt = pd.to_datetime(f"{d}/{m}/{y}", dayfirst=True, errors="coerce")
        except:
            pass
    if pd.notna(dt) and dt.year > 2400:
        dt = dt.replace(year=dt.year - 543)
    return dt

def load_gold():
    """โหลดข้อมูลทอง"""
    p = RAW / "gold_history.csv"
    if not p.exists():
        raise FileNotFoundError(f"[ERROR] gold_history.csv not found: {p}")

    df = pd.read_csv(p)
    df["date"] = df["date"].apply(parse_buddhist_date)
    df = df.dropna(subset=["date"])

    if "gold_bar_buy" in df.columns:
        val_col = "gold_bar_buy"
    elif "gold_sell" in df.columns:
        val_col = "gold_sell"
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        val_col = num_cols[-1]

    df = df[["date", val_col]].rename(columns={val_col: "gold"})
    df["gold"] = pd.to_numeric(df["gold"], errors="coerce")
    df = df.dropna(subset=["gold"]).drop_duplicates(subset=["date"], keep="last")
    return df.sort_values("date")

def load_fx():
    """โหลด USD/THB"""
    p = RAW / "exchange_rate.csv"
    df = pd.read_csv(p)
    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"].astype(str) + "-01", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    rate_col = (
        "mid_rate" if "mid_rate" in df.columns
        else ("selling" if "selling" in df.columns else "buying_transfer")
    )

    out = df.dropna(subset=["date"])[["date", rate_col]]
    out = out.rename(columns={rate_col: "fx"}).sort_values("date")
    out["fx"] = pd.to_numeric(out["fx"], errors="coerce")
    return out.dropna().drop_duplicates("date")

def load_cpi():
    """โหลด CPI"""
    p = RAW / "CPI_clean_for_supabase.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    val_col = "cpi_index" if "cpi_index" in df.columns else "value"
    df["cpi"] = pd.to_numeric(df[val_col], errors="coerce")
    return df[["date", "cpi"]].dropna().drop_duplicates("date").sort_values("date")

def load_oil():
    """โหลดน้ำมัน"""
    p = RAW / "petroleum_data.csv"
    df = pd.read_csv(p)
    if "period" in df.columns:
        df["date"] = pd.to_datetime(df["period"].astype(str) + "-01", errors="coerce")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    val_col = "value" if "value" in df.columns else df.select_dtypes(np.number).columns[-1]
    df["oil"] = pd.to_numeric(df[val_col], errors="coerce")
    out = df.dropna(subset=["date", "oil"])
    out = out.groupby("date", as_index=False)["oil"].mean()
    return out.sort_values("date").drop_duplicates("date")

def load_set():
    """โหลด SET"""
    p = RAW / "set_index.csv"
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    col = "Close" if "Close" in df.columns else df.select_dtypes(np.number).columns[0]
    df["set"] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "set"]].dropna().drop_duplicates("date").sort_values("date")

def load_btc(path):
    """โหลด Bitcoin"""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    
    df = pd.read_csv(p)
    date_col = "Date" if "Date" in df.columns else "date"
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    
    if "btc_price" in df.columns:
        price_col = "btc_price"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        price_col = "close"
    
    df["btc"] = pd.to_numeric(df[price_col], errors="coerce")
    return df[["date", "btc"]].dropna().drop_duplicates("date").sort_values("date")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc", type=str, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--roll", type=int, default=7)
    ap.add_argument("--minp", type=int, default=3)
    args = ap.parse_args()

    print(f"[INFO] RAW = {RAW}")

    gold = load_gold()
    fx = load_fx()
    cpi = load_cpi()
    oil = load_oil()
    seti = load_set()
    btc = load_btc(args.btc)

    all_data = [gold, fx, cpi, oil, seti] + ([btc] if btc is not None else [])

    start = min(df["date"].min() for df in all_data)
    end = max(df["date"].max() for df in all_data)

    calendar = pd.DataFrame({"date": pd.bdate_range(start, end)})
    feat = calendar.copy()

    for df in [gold, fx, cpi, oil, seti]:
        feat = feat.merge(df, on="date", how="left")

    if btc is not None:
        feat = feat.merge(btc, on="date", how="left")

    # Fill forward
    feat = feat.sort_values("date")
    for col in feat.columns:
        if col != "date":
            feat[col] = feat[col].ffill().bfill()

    # gold_next
    feat["gold_next"] = feat["gold"].shift(-1)

    # Create features - ใช้ชื่อที่ถูกต้อง
    vars_all = [c for c in ["gold", "fx", "cpi", "oil", "set", "btc"] if c in feat.columns]

    for col in vars_all:
        feat[f"{col}_lag1"] = feat[col].shift(1)
        feat[f"{col}_lag3"] = feat[col].shift(3)
        feat[f"{col}_roll7_mean"] = feat[col].rolling(args.roll, min_periods=args.minp).mean()  # ✅ แก้ที่นี่
        feat[f"{col}_pct_change"] = feat[col].pct_change()  # ✅ แก้ที่นี่

    feat = feat.dropna().reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(args.out, index=False)

    print(f"[OK] Feature store saved: {args.out}")
    print(f"[OK] rows = {len(feat)}, cols = {len(feat.columns)}")
    
    # แสดงคอลัมน์ที่สร้าง
    feature_cols = [c for c in feat.columns if any(x in c for x in ['lag', 'roll', 'pct'])]
    print(f"[OK] Created {len(feature_cols)} features")

if __name__ == "__main__":
    main()
