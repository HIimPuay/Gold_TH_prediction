#!/usr/bin/env python3
"""
Daily Pipeline - Simple & Working Version
"""
import subprocess as sp
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent

def run(cmd):
    """รันคำสั่ง"""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    ret = sp.run(cmd, cwd=BASE)
    return ret.returncode == 0

def main():
    print("\n🚀 DAILY PIPELINE")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. ดึงข้อมูล
    print("📥 Step 1: Fetching data...")
    run(["python3", "ingest_gold.py"])
    
    # 2. Build feature store (ใช้ Python inline แทน)
    print("\n🏗️  Step 2: Building feature store...")
    sp.run(["python3", "-c", """
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path('data/raw')

# Gold - ใช้ gold_bar_buy
df_gold = pd.read_csv(RAW / 'gold_history.csv')
df_gold['date'] = pd.to_datetime(df_gold['datetime'], errors='coerce')
df_gold = df_gold.dropna(subset=['date'])
df_gold['gold'] = pd.to_numeric(df_gold['gold_bar_buy'].astype(str).str.replace(',', ''), errors='coerce')
df_gold = df_gold[['date', 'gold']].dropna().drop_duplicates('date').sort_values('date')

# Other data
df_fx = pd.read_csv(RAW / 'exchange_rate.csv')
if 'period' in df_fx.columns:
    df_fx['date'] = pd.to_datetime(df_fx['period'].astype(str) + '-01', errors='coerce')
df_fx['fx'] = pd.to_numeric(df_fx['mid_rate'] if 'mid_rate' in df_fx.columns else df_fx['selling'], errors='coerce')
df_fx = df_fx[['date', 'fx']].dropna().drop_duplicates('date').sort_values('date')

df_cpi = pd.read_csv(RAW / 'CPI_clean_for_supabase.csv')
df_cpi['date'] = pd.to_datetime(df_cpi['date'], errors='coerce')
df_cpi['cpi'] = pd.to_numeric(df_cpi['cpi_index'] if 'cpi_index' in df_cpi.columns else df_cpi['value'], errors='coerce')
df_cpi = df_cpi[['date', 'cpi']].dropna().drop_duplicates('date').sort_values('date')

df_oil = pd.read_csv(RAW / 'petroleum_data.csv')
if 'period' in df_oil.columns:
    df_oil['date'] = pd.to_datetime(df_oil['period'].astype(str) + '-01', errors='coerce')
df_oil['oil'] = pd.to_numeric(df_oil['value'], errors='coerce')
df_oil = df_oil.groupby('date')['oil'].mean().reset_index().sort_values('date')

df_set = pd.read_csv(RAW / 'set_index.csv')
df_set['date'] = pd.to_datetime(df_set['date'], errors='coerce')
df_set['set'] = pd.to_numeric(df_set['Close'], errors='coerce')
df_set = df_set[['date', 'set']].dropna().drop_duplicates('date').sort_values('date')

# Calendar
start = df_gold['date'].min()
end = df_gold['date'].max()
calendar = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

# Merge
feat = calendar.copy()
for df in [df_gold, df_fx, df_cpi, df_oil, df_set]:
    feat = feat.merge(df, on='date', how='left')

# Forward fill
feat = feat.sort_values('date')
for col in ['gold', 'fx', 'cpi', 'oil', 'set']:
    feat[col] = feat[col].ffill().bfill()

# Features
feat['gold_next'] = feat['gold'].shift(-1)
for var in ['gold', 'fx', 'cpi', 'oil', 'set']:
    feat[f'{var}_lag1'] = feat[var].shift(1)
    feat[f'{var}_lag3'] = feat[var].shift(3)
    feat[f'{var}_roll7_mean'] = feat[var].rolling(7, min_periods=3).mean()
    feat[f'{var}_pct_change'] = feat[var].pct_change()

# Clean & Save
feat_clean = feat.dropna(subset=['gold', 'gold_lag1', 'gold_lag3'])
Path('data/Feature_store').mkdir(parents=True, exist_ok=True)
feat_clean.to_csv('data/Feature_store/feature_store.csv', index=False)
print(f'✅ Feature store: {len(feat_clean)} rows, latest = {feat_clean.iloc[-1]["date"].date()}')
    """], cwd=BASE)
    
    # 3. Train (Sunday only)
    if datetime.now().weekday() == 6:
        print("\n🎓 Step 3: Training model (Sunday)...")
        run(["python3", "model/train_model.py"])
    
    # 4. Predict
    print("\n🔮 Step 4: Making predictions...")
    run(["python3", "model/predict_gold.py", "--days", "7", "--save"])
    
    # 5. Dashboard
    print("\n📊 Step 5: Dashboard...")
    run(["python3", "dashboard.py"])
    
    print("\n✅ PIPELINE COMPLETED!\n")

if __name__ == "__main__":
    main()
