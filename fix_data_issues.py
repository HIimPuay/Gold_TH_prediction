#!/usr/bin/env python3
"""
fix_data_issues.py - ซ่อมแซมปัญหาข้อมูลที่พบบ่อย

ปัญหาที่แก้:
1. วันที่ไม่ตรง (พ.ศ. vs ค.ศ.)
2. ราคาที่เป็น string หรือมี comma
3. ข้อมูลซ้ำ
4. Missing values
5. Outliers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# ==================== CONFIG ====================
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"

# ==================== HELPERS ====================
THAI_MONTHS = {
    "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", 
    "เมษายน": "04", "พฤษภาคม": "05", "มิถุนายน": "06",
    "กรกฎาคม": "07", "สิงหาคม": "08", "กันยายน": "09",
    "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12"
}

def parse_thai_date(date_str):
    """แปลงวันที่ไทย (พ.ศ.) เป็น datetime"""
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).strip()
    
    # รูปแบบ: dd/mm/yyyy (พ.ศ.)
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
        try:
            day, month, year = date_str.split('/')
            year = int(year)
            if year > 2400:  # พ.ศ.
                year -= 543
            return pd.to_datetime(f"{year:04d}-{int(month):02d}-{int(day):02d}")
        except:
            pass
    
    # รูปแบบ: dd เดือนไทย yyyy
    parts = date_str.split()
    if len(parts) == 3:
        day = parts[0]
        month_th = parts[1]
        year = parts[2]
        
        if month_th in THAI_MONTHS:
            month = THAI_MONTHS[month_th]
            year = int(year)
            if year > 2400:
                year -= 543
            try:
                return pd.to_datetime(f"{year:04d}-{month}-{int(day):02d}")
            except:
                pass
    
    # ลอง parse ตรง ๆ
    try:
        dt = pd.to_datetime(date_str)
        if dt.year > 2400:
            dt = dt.replace(year=dt.year - 543)
        return dt
    except:
        return pd.NaT

def clean_numeric(value):
    """ทำความสะอาดค่าตัวเลข"""
    if pd.isna(value):
        return np.nan
    
    # แปลงเป็น string
    s = str(value).strip()
    
    # ลบ comma, space, บาท, THB, etc.
    s = re.sub(r'[,\s฿บาทTHB]', '', s)
    
    # แปลงเป็นตัวเลข
    try:
        return float(s)
    except:
        return np.nan

def detect_outliers(series, method='iqr', threshold=3):
    """หา outliers"""
    series = series.dropna()
    
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        return (series < lower) | (series > upper)
    
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold
    
    return pd.Series(False, index=series.index)

# ==================== FIXERS ====================
def fix_gold_data():
    """ซ่อมแซมข้อมูลทอง"""
    print("\n" + "=" * 60)
    print("🔧 Fixing Gold Data")
    print("=" * 60)
    
    gold_file = RAW_DIR / "gold_history.csv"
    if not gold_file.exists():
        print("❌ Gold file not found")
        return False
    
    try:
        df = pd.read_csv(gold_file)
        original_len = len(df)
        print(f"📊 Original data: {original_len} rows")
        
        # 1. Fix date
        print("\n1️⃣  Fixing dates...")
        if 'date' in df.columns:
            df['date'] = df['date'].apply(parse_thai_date)
        if 'datetime' in df.columns:
            df['datetime'] = df['datetime'].apply(parse_thai_date)
        elif 'date' in df.columns:
            df['datetime'] = df['date']
        
        # 2. Fix price columns
        print("2️⃣  Fixing prices...")
        price_cols = ['gold_buy', 'gold_sell', 'gold_bar_buy', 'gold_bar_sell']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
                
                # ตรวจสอบช่วงราคา (ทองคำไทยควรอยู่ระหว่าง 10,000-100,000)
                valid_range = (df[col] >= 10000) & (df[col] <= 100000)
                invalid_count = (~valid_range & df[col].notna()).sum()
                if invalid_count > 0:
                    print(f"   ⚠️  {col}: {invalid_count} values out of valid range")
                    df.loc[~valid_range, col] = np.nan
        
        # 3. Remove invalid rows
        print("3️⃣  Removing invalid rows...")
        df = df.dropna(subset=['datetime'])
        
        # ต้องมีราคาอย่างน้อย 1 คอลัมน์
        has_any_price = df[price_cols].notna().any(axis=1)
        df = df[has_any_price]
        
        # 4. Remove duplicates
        print("4️⃣  Removing duplicates...")
        df = df.sort_values('datetime')
        duplicates = df.duplicated(subset=['datetime'], keep='last')
        dup_count = duplicates.sum()
        if dup_count > 0:
            print(f"   Found {dup_count} duplicates")
        df = df[~duplicates]
        
        # 5. Detect outliers
        print("5️⃣  Detecting outliers...")
        for col in price_cols:
            if col in df.columns and df[col].notna().sum() > 10:
                outliers = detect_outliers(df[col])
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    print(f"   ⚠️  {col}: {outlier_count} outliers detected")
                    # แสดง outliers
                    if outlier_count < 10:
                        print(f"      Values: {df.loc[outliers, col].tolist()}")
        
        # 6. Sort and save
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Backup original
        backup_file = gold_file.with_suffix('.backup.csv')
        if not backup_file.exists():
            pd.read_csv(gold_file).to_csv(backup_file, index=False)
            print(f"💾 Backup saved: {backup_file}")
        
        # Save fixed data
        df.to_csv(gold_file, index=False)
        
        print(f"\n✅ Fixed data: {len(df)} rows (removed {original_len - len(df)})")
        print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def fix_exchange_rate():
    """ซ่อมแซมข้อมูลอัตราแลกเปลี่ยน"""
    print("\n" + "=" * 60)
    print("🔧 Fixing Exchange Rate Data")
    print("=" * 60)
    
    fx_file = RAW_DIR / "exchange_rate.csv"
    if not fx_file.exists():
        print("❌ Exchange rate file not found")
        return False
    
    try:
        df = pd.read_csv(fx_file)
        original_len = len(df)
        print(f"📊 Original data: {original_len} rows")
        
        # Fix date
        if 'period' in df.columns:
            df['date'] = pd.to_datetime(df['period'].astype(str) + '-01', errors='coerce')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Fix rate columns
        rate_cols = ['mid_rate', 'selling', 'buying_transfer', 'buying_sight']
        for col in rate_cols:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
                
                # เช็คช่วง USD/THB (ควรอยู่ระหว่าง 20-50)
                valid_range = (df[col] >= 20) & (df[col] <= 50)
                invalid = (~valid_range & df[col].notna()).sum()
                if invalid > 0:
                    print(f"   ⚠️  {col}: {invalid} values out of range")
                    df.loc[~valid_range, col] = np.nan
        
        # Remove invalid and duplicates
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Save
        df.to_csv(fx_file, index=False)
        
        print(f"✅ Fixed data: {len(df)} rows")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_feature_store_quality():
    """ตรวจสอบคุณภาพ feature store"""
    print("\n" + "=" * 60)
    print("🔍 Checking Feature Store Quality")
    print("=" * 60)
    
    fs_path = BASE / "data" / "Feature_store" / "feature_store.csv"
    if not fs_path.exists():
        print("⚠️  Feature store doesn't exist yet")
        return True
    
    try:
        df = pd.read_csv(fs_path, parse_dates=['date'])
        
        print(f"📊 Feature store: {len(df)} rows")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check missing values
        print("\n📋 Missing Values:")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            print("   ✅ No missing values")
        else:
            for col, count in missing.head(10).items():
                pct = count / len(df) * 100
                print(f"   {col}: {count} ({pct:.1f}%)")
        
        # Check gold price statistics
        if 'gold' in df.columns:
            print(f"\n💰 Gold Price Statistics:")
            print(f"   Mean:   {df['gold'].mean():.2f}")
            print(f"   Median: {df['gold'].median():.2f}")
            print(f"   Std:    {df['gold'].std():.2f}")
            print(f"   Min:    {df['gold'].min():.2f}")
            print(f"   Max:    {df['gold'].max():.2f}")
            
            # Check recent trend
            recent = df.tail(30)
            trend = recent['gold'].iloc[-1] - recent['gold'].iloc[0]
            print(f"   30-day change: {'+' if trend > 0 else ''}{trend:.2f} THB")
        
        # Check prediction readiness
        print(f"\n🎯 Prediction Readiness:")
        required_features = [
            'gold', 'fx', 'cpi', 'oil', 'set',
            'gold_lag1', 'gold_lag3', 'gold_roll7', 'gold_pct'
        ]
        
        ready = True
        for feat in required_features:
            if feat not in df.columns:
                print(f"   ❌ Missing: {feat}")
                ready = False
            elif df[feat].isna().all():
                print(f"   ❌ All NaN: {feat}")
                ready = False
            elif df[feat].isna().sum() > len(df) * 0.5:
                pct = df[feat].isna().sum() / len(df) * 100
                print(f"   ⚠️  {feat}: {pct:.1f}% missing")
            else:
                print(f"   ✅ {feat}: OK")
        
        if ready:
            print("\n✅ Feature store is ready for prediction")
        else:
            print("\n⚠️  Feature store needs fixing")
        
        return ready
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# ==================== MAIN ====================
def main():
    print("=" * 60)
    print("🔧 DATA REPAIR TOOL")
    print("=" * 60)
    
    print(f"\n📁 Working directory: {BASE}")
    print(f"📁 Raw data directory: {RAW_DIR}")
    
    # Fix each data source
    fix_gold_data()
    fix_exchange_rate()
    
    # Check feature store
    check_feature_store_quality()
    
    print("\n" + "=" * 60)
    print("✅ Data repair completed")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("1. Run: python3 daily_pipeline_fixed.py")
    print("2. Check logs for any remaining issues")
    print("3. Verify predictions with: python3 model/predict_gold.py")

if __name__ == "__main__":
    main()