#!/usr/bin/env python3
"""
data_alignment_fixed.py - รวมข้อมูลทั้งหมด (แก้ไขแล้ว)
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RAW_DIR = BASE / "data" / "raw"
ALIGNED_DIR = BASE / "data" / "aligned"

# Input files - ใช้ไฟล์ที่มีจริง
GOLD_FILE = RAW_DIR / "gold_history.csv"
FX_FILE = RAW_DIR / "exchange_rate.csv"  # ใช้ exchange_rate.csv แทน
CPI_FILE = RAW_DIR / "CPI_clean_for_supabase.csv"
OIL_FILE = RAW_DIR / "petroleum_data.csv"
SET_FILE = RAW_DIR / "set_index.csv"
BTC_FILE = RAW_DIR / "bitcoin_history.csv"

OUTPUT_FILE = ALIGNED_DIR / "aligned_daily.csv"

def load_gold():
    """โหลดข้อมูลทอง"""
    df = pd.read_csv(GOLD_FILE)
    if 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime']).dt.date
    else:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    # เลือกคอลัมน์ราคา
    if 'gold_sell' in df.columns:
        df['gold'] = df['gold_sell']
    elif 'gold_bar_sell' in df.columns:
        df['gold'] = df['gold_bar_sell']
    
    return df[['date', 'gold']].drop_duplicates('date')

def load_fx():
    """โหลด USD/THB - ใช้ exchange_rate.csv"""
    df = pd.read_csv(FX_FILE)
    
    if 'period' in df.columns:
        df['date'] = pd.to_datetime(df['period'].astype(str) + '-01').dt.date
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        df['date'] = pd.to_datetime(df['Date']).dt.date
    
    # เลือกคอลัมน์อัตรา
    if 'mid_rate' in df.columns:
        df['fx'] = df['mid_rate']
    elif 'selling' in df.columns:
        df['fx'] = df['selling']
    elif 'Price' in df.columns:
        df['fx'] = df['Price']
    
    return df[['date', 'fx']].drop_duplicates('date')

def load_cpi():
    """โหลด CPI"""
    df = pd.read_csv(CPI_FILE)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    if 'cpi_index' in df.columns:
        df['cpi'] = df['cpi_index']
    else:
        df['cpi'] = df['value']
    
    return df[['date', 'cpi']].drop_duplicates('date')

def load_oil():
    """โหลดราคาน้ำมัน"""
    df = pd.read_csv(OIL_FILE)
    
    if 'period' in df.columns:
        df['date'] = pd.to_datetime(df['period'].astype(str) + '-01').dt.date
    else:
        df['date'] = pd.to_datetime(df['date']).dt.date
    
    df['oil'] = df['value']
    df = df.groupby('date', as_index=False)['oil'].mean()
    
    return df[['date', 'oil']].drop_duplicates('date')

def load_set():
    """โหลด SET Index"""
    df = pd.read_csv(SET_FILE)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    if 'Close' in df.columns:
        df['set'] = df['Close']
    else:
        df['set'] = df['close']
    
    return df[['date', 'set']].drop_duplicates('date')

def load_bitcoin():
    """โหลด Bitcoin"""
    if not BTC_FILE.exists():
        print("⚠️  Bitcoin data not found")
        return pd.DataFrame(columns=['date', 'btc'])
    
    df = pd.read_csv(BTC_FILE)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    if 'btc_price' in df.columns:
        df['btc'] = df['btc_price']
    elif 'Close' in df.columns:
        df['btc'] = df['Close']
    
    return df[['date', 'btc']].drop_duplicates('date')

def main():
    print("🔗 Data Alignment (Fixed)")
    print("=" * 60)
    
    print("📊 Loading data...")
    df_gold = load_gold()
    df_fx = load_fx()
    df_cpi = load_cpi()
    df_oil = load_oil()
    df_set = load_set()
    df_btc = load_bitcoin()
    
    print(f"   Gold:    {len(df_gold)} days")
    print(f"   FX:      {len(df_fx)} days")
    print(f"   CPI:     {len(df_cpi)} months")
    print(f"   Oil:     {len(df_oil)} days")
    print(f"   SET:     {len(df_set)} days")
    print(f"   Bitcoin: {len(df_btc)} days")
    
    # Merge
    df = df_gold.copy()
    df = df.merge(df_fx, on='date', how='outer')
    df = df.merge(df_cpi, on='date', how='outer')
    df = df.merge(df_oil, on='date', how='outer')
    df = df.merge(df_set, on='date', how='outer')
    
    if not df_btc.empty:
        df = df.merge(df_btc, on='date', how='outer')
    else:
        df['btc'] = None
    
    # Sort and fill
    df = df.sort_values('date').reset_index(drop=True)
    df = df.ffill()
    df = df.dropna(subset=['gold'])
    
    print(f"\n✅ Merged: {len(df)} rows")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
