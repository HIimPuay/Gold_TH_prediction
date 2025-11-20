#!/usr/bin/env python3


import subprocess as sp
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# ==================== CONFIG ====================
ICT = timezone(timedelta(hours=7))
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"
FEATURE_STORE_DIR = BASE / "data" / "Feature_store"
FEATURE_STORE_PATH = FEATURE_STORE_DIR / "feature_store.csv"

# ==================== LOGGING ====================
def log(msg, level="INFO"):
    timestamp = datetime.now(ICT).strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
    }.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")

# ==================== VALIDATION ====================
def validate_raw_data():
    """ตรวจสอบไฟล์ raw data ที่จำเป็น"""
    log("Validating raw data files...")
    
    required_files = {
        "gold_history.csv": "Gold prices",
        "exchange_rate.csv": "Exchange rates", 
        "CPI_clean_for_supabase.csv": "CPI data",
        "petroleum_data.csv": "Oil prices",
        "set_index.csv": "SET index",
        "bitcoin_history.csv": "Bitcoin prices (optional)"
    }
    
    missing = []
    for filename, description in required_files.items():
        filepath = RAW_DIR / filename
        if not filepath.exists():
            if "optional" not in description.lower():
                missing.append(f"{filename} ({description})")
                log(f"Missing: {filename}", "ERROR")
            else:
                log(f"Optional file not found: {filename}", "WARNING")
        else:
            # ตรวจสอบว่าไฟล์ไม่ว่าง
            try:
                df = pd.read_csv(filepath)
                if len(df) == 0:
                    missing.append(f"{filename} (empty file)")
                    log(f"Empty file: {filename}", "ERROR")
                else:
                    log(f"Found {filename}: {len(df)} rows", "SUCCESS")
            except Exception as e:
                missing.append(f"{filename} (read error: {e})")
                log(f"Error reading {filename}: {e}", "ERROR")
    
    if missing:
        log(f"Missing or invalid files: {', '.join(missing)}", "ERROR")
        return False
    
    log("All required raw data files are valid", "SUCCESS")
    return True

def clean_gold_data():
    """ทำความสะอาดข้อมูลทอง"""
    log("Cleaning gold data...")
    
    gold_file = RAW_DIR / "gold_history.csv"
    if not gold_file.exists():
        log("Gold file not found", "ERROR")
        return False
    
    try:
        df = pd.read_csv(gold_file)
        
        # แปลง datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            log("No date column found in gold data", "ERROR")
            return False
        
        # ตรวจสอบและแปลงราคาทอง
        price_columns = ['gold_sell', 'gold_bar_sell', 'gold_buy', 'gold_bar_buy']
        found_price_col = None
        
        for col in price_columns:
            if col in df.columns:
                # ทำความสะอาด: ลบ comma, space
                df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if found_price_col is None:
                    found_price_col = col
        
        if found_price_col is None:
            log("No price column found in gold data", "ERROR")
            return False
        
        # กรองข้อมูลที่ valid
        df = df.dropna(subset=['datetime', found_price_col])
        df = df[df[found_price_col] > 0]  # ราคาต้องมากกว่า 0
        
        # ลบข้อมูลซ้ำ
        df = df.sort_values('datetime')
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # บันทึกกลับ
        df.to_csv(gold_file, index=False)
        
        log(f"Gold data cleaned: {len(df)} valid rows", "SUCCESS")
        return True
        
    except Exception as e:
        log(f"Error cleaning gold data: {e}", "ERROR")
        return False

def validate_feature_store():
    """ตรวจสอบ feature store"""
    log("Validating feature store...")
    
    if not FEATURE_STORE_PATH.exists():
        log("Feature store not found (will be created)", "WARNING")
        return True  # OK ถ้ายังไม่มี
    
    try:
        df = pd.read_csv(FEATURE_STORE_PATH, parse_dates=['date'])
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = [
            'date', 'gold', 'fx', 'cpi', 'oil', 'set', 'gold_next',
            'gold_lag1', 'gold_lag3', 'gold_roll7', 'gold_pct'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            log(f"Missing columns in feature store: {missing_cols}", "ERROR")
            return False
        
        # ตรวจสอบ NaN
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.sum() > 0:
            log(f"Found NaN values in feature store:", "WARNING")
            for col, count in nan_counts[nan_counts > 0].items():
                log(f"  {col}: {count} NaN values", "WARNING")
        
        # ตรวจสอบช่วงวันที่
        latest_date = df['date'].max()
        days_old = (datetime.now() - latest_date).days
        
        if days_old > 7:
            log(f"Feature store is {days_old} days old (last: {latest_date.date()})", "WARNING")
        else:
            log(f"Feature store is up to date (last: {latest_date.date()})", "SUCCESS")
        
        log(f"Feature store validated: {len(df)} rows", "SUCCESS")
        return True
        
    except Exception as e:
        log(f"Error validating feature store: {e}", "ERROR")
        return False

# ==================== PIPELINE STEPS ====================
def run_step(cmd, name, critical=True):
    """รันคำสั่งและจัดการ error"""
    log(f"Starting: {name}")
    
    try:
        ret = sp.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=300)
        
        if ret.returncode != 0:
            log(f"Failed: {name}", "ERROR")
            log(f"Error output: {ret.stderr}", "ERROR")
            
            if critical:
                log("Critical step failed, stopping pipeline", "ERROR")
                return False
            else:
                log("Non-critical step failed, continuing...", "WARNING")
                return True
        
        log(f"Completed: {name}", "SUCCESS")
        if ret.stdout:
            print(ret.stdout)
        
        return True
        
    except sp.TimeoutExpired:
        log(f"Timeout: {name} (>5 minutes)", "ERROR")
        return False if critical else True
    except Exception as e:
        log(f"Exception in {name}: {e}", "ERROR")
        return False if critical else True

# ==================== MAIN PIPELINE ====================
def main():
    log("=" * 70)
    log("DAILY PIPELINE START (FIXED VERSION)")
    log("=" * 70)
    
    pipeline_success = True
    
    # Step 0: Validate raw data
    log("\n📋 Step 0: Validate Raw Data")
    if not validate_raw_data():
        log("Raw data validation failed", "ERROR")
        pipeline_success = False
    
    # Step 1: ดึงข้อมูล Gold + Bitcoin
    log("\n📥 Step 1: Fetch Gold & Bitcoin Data")
    if not run_step(
        ["python3", "ingest_gold.py"],
        "FETCH_GOLD_BITCOIN",
        critical=True
    ):
        pipeline_success = False
    
    # Step 1.5: Clean gold data
    log("\n🧹 Step 1.5: Clean Gold Data")
    if not clean_gold_data():
        pipeline_success = False
    
    # Step 2: Align data
    log("\n🔗 Step 2: Align Data")
    if not run_step(
        ["python3", "scripts/data_alignment_steps_btc.py"],
        "ALIGN_DATA",
        critical=True
    ):
        pipeline_success = False
    
    # Step 3: Build feature store
    log("\n🏗️  Step 3: Build Feature Store")
    if not run_step(
        ["python3", "scripts/build_feature_store_btc.py"],
        "BUILD_FEATURE_STORE",
        critical=True
    ):
        pipeline_success = False
    
    # Step 4: Validate feature store
    log("\n✅ Step 4: Validate Feature Store")
    if not run_step(
        ["python3", "scripts/validate_feature_store_btc.py",
         "--path", str(FEATURE_STORE_PATH)],
        "VALIDATE_FEATURE_STORE",
        critical=False
    ):
        log("Feature store validation had issues, but continuing...", "WARNING")
    
    # Step 5: Train model (Sunday only)
    if datetime.now(ICT).weekday() == 6:  # Sunday
        log("\n🎓 Step 5: Train Model (Sunday)")
        if not run_step(
            ["python3", "model/train_model.py", "--plot"],
            "TRAIN_MODEL",
            critical=False
        ):
            log("Model training failed, but continuing...", "WARNING")
    else:
        log("\n⏭️  Step 5: Skip Training (not Sunday)")
    
    # Step 6: Predict
    log("\n🔮 Step 6: Make Predictions")
    if not run_step(
        ["python3", "model/predict_gold.py", "--days", "7", "--save"],
        "PREDICT",
        critical=False
    ):
        log("Prediction failed", "WARNING")
    
    # Step 7: Dashboard
    log("\n📊 Step 7: Generate Dashboard")
    if not run_step(
        ["python3", "dashboard.py"],
        "DASHBOARD",
        critical=False
    ):
        log("Dashboard generation failed", "WARNING")
    
    # Final summary
    log("\n" + "=" * 70)
    if pipeline_success:
        log("PIPELINE COMPLETED SUCCESSFULLY", "SUCCESS")
    else:
        log("PIPELINE COMPLETED WITH ERRORS", "ERROR")
    log("=" * 70)
    
    return 0 if pipeline_success else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("Pipeline interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"Unexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)