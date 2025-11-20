#!/usr/bin/env python3
"""
data_alignment_steps_btc.py - รวมข้อมูลทั้งหมด (รวม Bitcoin)
"""
import pandas as pd
from pathlib import Path

# ==== PATH CONFIG ====
BASE = Path("/Users/nichanun/Desktop/DSDN")
RAW_DIR = BASE / "data" / "raw"
ALIGNED_DIR = BASE / "data" / "aligned"

# Input files
GOLD_FILE = RAW_DIR / "gold_history.csv"
USD_FILE = RAW_DIR / "USD_THB_Historical Data.csv"
CPI_FILE = RAW_DIR / "CPI_Thailand_Monthly.csv"
OIL_FILE = RAW_DIR / "Brent_Oil_Futures_Historical_Data.csv"
SET_FILE = RAW_DIR / "SET Index Historical Data.csv"
BTC_FILE = RAW_DIR / "bitcoin_history.csv"  # ← เพิ่ม Bitcoin

# Output
OUTPUT_FILE = ALIGNED_DIR / "aligned_daily.csv"

def load_gold():
    """โหลดข้อมูลทอง"""
    df = pd.read_csv(GOLD_FILE)
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['gold'] = df['gold_sell']
    return df[['date', 'gold']].drop_duplicates('date')

def load_usd_thb():
    """โหลด USD/THB"""
    df = pd.read_csv(USD_FILE)
    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['fx'] = df['Price']
    return df[['date', 'fx']].drop_duplicates('date')

def load_cpi():
    """โหลด CPI (รายเดือน → forward fill รายวัน)"""
    df = pd.read_csv(CPI_FILE)
    df['date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d').dt.date
    df['cpi'] = df['CPI']
    return df[['date', 'cpi']].drop_duplicates('date')

def load_oil():
    """โหลดราคาน้ำมัน"""
    df = pd.read_csv(OIL_FILE)
    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['oil'] = df['Price']
    return df[['date', 'oil']].drop_duplicates('date')

def load_set():
    """โหลด SET Index"""
    df = pd.read_csv(SET_FILE)
    df['date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y').dt.date
    df['set'] = df['Price']
    return df[['date', 'set']].drop_duplicates('date')

def load_bitcoin():
    """โหลด Bitcoin (BTC/THB)"""
    if not BTC_FILE.exists():
        print("⚠️  Bitcoin data not found, skipping...")
        return pd.DataFrame(columns=['date', 'btc'])
    
    df = pd.read_csv(BTC_FILE)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['btc'] = df['btc_price']
    return df[['date', 'btc']].drop_duplicates('date')

def merge_all():
    """รวมข้อมูลทั้งหมด"""
    print("📊 Loading data...")
    
    df_gold = load_gold()
    df_fx = load_usd_thb()
    df_cpi = load_cpi()
    df_oil = load_oil()
    df_set = load_set()
    df_btc = load_bitcoin()
    
    print(f"   Gold:    {len(df_gold)} days")
    print(f"   USD/THB: {len(df_fx)} days")
    print(f"   CPI:     {len(df_cpi)} months")
    print(f"   Oil:     {len(df_oil)} days")
    print(f"   SET:     {len(df_set)} days")
    print(f"   Bitcoin: {len(df_btc)} days")
    
    # Merge ทั้งหมด (outer join)
    df = df_gold.copy()
    df = df.merge(df_fx, on='date', how='outer')
    df = df.merge(df_cpi, on='date', how='outer')
    df = df.merge(df_oil, on='date', how='outer')
    df = df.merge(df_set, on='date', how='outer')
    
    # Merge Bitcoin (ถ้ามี)
    if not df_btc.empty:
        df = df.merge(df_btc, on='date', how='outer')
    else:
        df['btc'] = None
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Forward fill (CPI และวันหยุด)
    df = df.ffill()
    
    # Drop rows with missing gold (target variable)
    df = df.dropna(subset=['gold'])
    
    print(f"\n✅ Merged: {len(df)} rows")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Missing values:")
    print(df.isnull().sum())
    
    return df

def main():
    print("🔗 Data Alignment Pipeline")
    print("=" * 60)
    
    df = merge_all()
    
    # Save
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n💾 Saved to: {OUTPUT_FILE}")
    print(f"   Columns: {', '.join(df.columns)}")

if __name__ == "__main__":
    main()