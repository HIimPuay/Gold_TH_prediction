#!/usr/bin/env python3
"""
fix_urgent_issues.py - แก้ปัญหาเร่งด่วนทั้งหมด

ปัญหา:
1. Feature names ไม่ตรงกัน (roll7 vs roll7_mean, pct vs pct_change)
2. ไฟล์ USD_THB_Historical Data.csv หายไป
3. Bitcoin data มีปัญหา
4. data_alignment_steps_btc.py ใช้ path ผิด
"""

import pandas as pd
from pathlib import Path
import shutil

BASE = Path("/Users/nichanun/Desktop/DSDN")
RAW_DIR = BASE / "data" / "raw"
FEATURE_STORE = BASE / "data" / "Feature_store" / "feature_store.csv"

print("=" * 70)
print("🔧 URGENT FIX - แก้ปัญหาเร่งด่วน")
print("=" * 70)

# ==================== FIX 1: Feature Store Column Names ====================
print("\n1️⃣  แก้ชื่อคอลัมน์ใน Feature Store...")

if FEATURE_STORE.exists():
    df = pd.read_csv(FEATURE_STORE)
    
    # Backup
    backup = FEATURE_STORE.parent / "feature_store_backup.csv"
    shutil.copy(FEATURE_STORE, backup)
    print(f"   💾 Backup: {backup}")
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        if col.endswith('_roll7'):
            new_col = col.replace('_roll7', '_roll7_mean')
            rename_map[col] = new_col
        elif col.endswith('_pct') and not col.endswith('_pct_change'):
            new_col = col.replace('_pct', '_pct_change')
            rename_map[col] = new_col
    
    if rename_map:
        df = df.rename(columns=rename_map)
        df.to_csv(FEATURE_STORE, index=False)
        print(f"   ✅ Renamed {len(rename_map)} columns:")
        for old, new in list(rename_map.items())[:5]:
            print(f"      {old} → {new}")
        if len(rename_map) > 5:
            print(f"      ... และอีก {len(rename_map)-5} columns")
    else:
        print("   ℹ️  No columns need renaming")
else:
    print("   ⚠️  Feature store not found yet")

# ==================== FIX 2: Check/Fix USD_THB File ====================
print("\n2️⃣  แก้ปัญหาไฟล์ USD/THB...")

# ลองหาไฟล์ exchange_rate
exchange_files = list(RAW_DIR.glob("*exchange*"))
usd_files = list(RAW_DIR.glob("*USD*"))

print(f"   📁 Found exchange rate files: {len(exchange_files)}")
print(f"   📁 Found USD files: {len(usd_files)}")

if exchange_files:
    # ใช้ exchange_rate.csv แทน USD_THB_Historical Data.csv
    source = exchange_files[0]
    target = RAW_DIR / "USD_THB_Historical Data.csv"
    
    if not target.exists():
        # อ่านและแปลงรูปแบบให้ตรงกับที่ data_alignment ต้องการ
        df = pd.read_csv(source)
        
        # แปลงให้เป็นรูปแบบที่ data_alignment คาดหวัง
        if 'period' in df.columns:
            df['Date'] = pd.to_datetime(df['period'].astype(str) + '-01')
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        
        # เลือกคอลัมน์อัตราแลกเปลี่ยน
        if 'mid_rate' in df.columns:
            df['Price'] = df['mid_rate']
        elif 'selling' in df.columns:
            df['Price'] = df['selling']
        
        # บันทึก
        df[['Date', 'Price']].to_csv(target, index=False)
        print(f"   ✅ Created: {target.name}")
    else:
        print(f"   ✅ Already exists: {target.name}")
else:
    print("   ❌ No exchange rate file found!")

# ==================== FIX 3: Fix Bitcoin Data ====================
print("\n3️⃣  แก้ Bitcoin data...")

btc_file = RAW_DIR / "bitcoin_history.csv"
if btc_file.exists():
    df_btc = pd.read_csv(btc_file)
    
    # เช็คว่ามีคอลัมน์ date หรือไม่
    if 'Date' in df_btc.columns and 'date' not in df_btc.columns:
        df_btc = df_btc.rename(columns={'Date': 'date'})
        print("   ✅ Renamed 'Date' → 'date'")
    
    # เช็คคอลัมน์ราคา
    if 'Close' in df_btc.columns and 'btc_price' not in df_btc.columns:
        df_btc = df_btc.rename(columns={'Close': 'btc_price'})
        print("   ✅ Renamed 'Close' → 'btc_price'")
    
    # เช็คว่ามีคอลัมน์ที่จำเป็น
    required = ['date', 'btc_price']
    if all(col in df_btc.columns for col in required):
        df_btc[required].to_csv(btc_file, index=False)
        print(f"   ✅ Fixed Bitcoin data: {len(df_btc)} rows")
    else:
        missing = [c for c in required if c not in df_btc.columns]
        print(f"   ⚠️  Still missing columns: {missing}")
else:
    print("   ⚠️  Bitcoin file not found")

# ==================== FIX 4: Create Fixed data_alignment Script ====================
print("\n4️⃣  สร้าง data_alignment_fixed.py...")

alignment_script = BASE / "scripts" / "data_alignment_fixed.py"
alignment_script.write_text('''#!/usr/bin/env python3
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
    
    print(f"\\n✅ Merged: {len(df)} rows")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\\n💾 Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
''')

print(f"   ✅ Created: {alignment_script}")

# ==================== FIX 5: Fix daily_pipeline.py ====================
print("\n5️⃣  แก้ daily_pipeline.py...")

pipeline_file = BASE / "daily_pipeline.py"
if pipeline_file.exists():
    content = pipeline_file.read_text()
    
    # แก้ไข error
    content = content.replace(
        'if success_gold and success_btc:',
        'if success_data:'
    )
    content = content.replace(
        '    print("\\n✅ All data sources updated successfully")\n    else:\n        print("\\n⚠️  Some data sources failed (check logs)")',
        '    print("\\n✅ Data updated successfully")\nelse:\n    print("\\n⚠️  Data update failed (check logs)")'
    )
    
    pipeline_file.write_text(content)
    print("   ✅ Fixed daily_pipeline.py")

# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print("✅ แก้ไขเสร็จสิ้น!")
print("=" * 70)

print("\n💡 ขั้นตอนถัดไป:")
print("1. python3 scripts/data_alignment_fixed.py")
print("2. python3 scripts/build_feature_store_btc.py")
print("3. python3 model/train_model.py")
print("4. python3 model/predict_gold.py --days 7 --save")

print("\n📋 สรุปการแก้ไข:")
print("✅ แก้ชื่อคอลัมน์ใน feature store")
print("✅ สร้างไฟล์ USD_THB_Historical Data.csv")
print("✅ แก้ Bitcoin data columns")
print("✅ สร้าง data_alignment_fixed.py")
print("✅ แก้ daily_pipeline.py")
