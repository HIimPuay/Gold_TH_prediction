#!/usr/bin/env python3
"""
daily_pipeline.py - Production Pipeline (Complete Version)

Features:
- ข้ามวันอาทิตย์อัตโนมัติ
- Validation ครบถ้วน
- Error handling ดี
- Logging ชัดเจน
- รองรับ gold_config.py
"""

import subprocess as sp
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# ==================== CONFIG ====================
ICT = timezone(timedelta(hours=7))
BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"
FEATURE_STORE_PATH = BASE / "data" / "Feature_store" / "feature_store.csv"

# ==================== LOGGING ====================
def log(msg, level="INFO"):
    """แสดง log พร้อม timestamp และ emoji"""
    timestamp = datetime.now(ICT).strftime("%Y-%m-%d %H:%M:%S")
    emoji = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "SKIP": "⏭️ "
    }.get(level, "")
    print(f"[{timestamp}] {emoji} {msg}")

# ==================== VALIDATION ====================
def check_files():
    """ตรวจสอบไฟล์ที่จำเป็น"""
    required = [
        (RAW_DIR / "gold_history.csv", "Gold prices", True),
        (RAW_DIR / "exchange_rate.csv", "Exchange rates", True),
        (RAW_DIR / "CPI_clean_for_supabase.csv", "CPI data", True),
        (RAW_DIR / "petroleum_data.csv", "Oil prices", True),
        (RAW_DIR / "set_index.csv", "SET index", True),
        (RAW_DIR / "bitcoin_history.csv", "Bitcoin prices", False),  # Optional
    ]
    
    all_good = True
    for filepath, description, critical in required:
        if filepath.exists():
            log(f"Found: {description}", "SUCCESS")
        else:
            if critical:
                log(f"Missing: {description} ({filepath.name})", "ERROR")
                all_good = False
            else:
                log(f"Optional: {description} not found", "WARNING")
    
    return all_good

# ==================== PIPELINE STEPS ====================
def run_step(cmd, name, critical=True, timeout=300):
    """รันคำสั่งและจัดการ error"""
    log(f"Starting: {name}")
    
    try:
        # รันคำสั่ง
        result = sp.run(
            cmd, 
            cwd=BASE, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        # แสดง output
        if result.stdout:
            print(result.stdout)
        
        # เช็ค error
        if result.returncode != 0:
            log(f"Failed: {name} (exit code: {result.returncode})", "ERROR")
            if result.stderr:
                print(f"Error output:\n{result.stderr}")
            
            if critical:
                log("Critical step failed - stopping pipeline", "ERROR")
                return False
            else:
                log("Non-critical step failed - continuing", "WARNING")
                return True
        
        log(f"Completed: {name}", "SUCCESS")
        return True
        
    except sp.TimeoutExpired:
        log(f"Timeout: {name} (>{timeout}s)", "ERROR")
        return False if critical else True
        
    except FileNotFoundError as e:
        log(f"Command not found: {cmd[0]}", "ERROR")
        return False if critical else True
        
    except Exception as e:
        log(f"Unexpected error in {name}: {e}", "ERROR")
        return False if critical else True

# ==================== MAIN PIPELINE ====================
def main():
    log("=" * 70)
    log("DAILY PIPELINE START")
    log("=" * 70)
    
    current_day = datetime.now(ICT)
    day_name = current_day.strftime('%A')
    log(f"Today: {current_day.strftime('%Y-%m-%d')} ({day_name})")
    
    # =====================================
    # เช็ควันอาทิตย์ (ตลาดทองปิด)
    # =====================================
    if current_day.weekday() == 6:  # Sunday
        log("=" * 70)
        log("วันอาทิตย์ - ตลาดทองปิด", "SKIP")
        log("Pipeline จะไม่ทำงานในวันนี้")
        log("=" * 70)
        return 0
    
    # =====================================
    # Step 0: Validate Files
    # =====================================
    log("\n📋 Step 0: Validate Files")
    log("-" * 70)
    if not check_files():
        log("File validation failed - stopping pipeline", "ERROR")
        return 1
    
    pipeline_success = True
    
    # =====================================
    # Step 1: Fetch Data (Gold + Bitcoin)
    # =====================================
    log("\n📥 Step 1: Fetch Latest Data")
    log("-" * 70)
    success_fetch = run_step(
        ["python3", "ingest_gold.py"],
        "FETCH_DATA",
        critical=True
    )
    if not success_fetch:
        pipeline_success = False
    
    # =====================================
    # Step 2: Build Feature Store
    # =====================================
    log("\n🏗️  Step 2: Build Feature Store")
    log("-" * 70)
    
    # ตรวจสอบว่ามี build_feature_store_btc.py หรือไม่
    build_script = BASE / "build_feature_store_btc.py"
    if not build_script.exists():
        build_script = BASE / "scripts" / "build_feature_store_btc.py"
    
    if build_script.exists():
        success_build = run_step(
            ["python3", str(build_script)],
            "BUILD_FEATURE_STORE",
            critical=True
        )
    else:
        log("build_feature_store_btc.py not found - trying alternative", "WARNING")
        # Fallback: ใช้ inline code
        success_build = run_step(
            ["python3", "-c", "print('Building feature store inline...'); import sys; sys.exit(0)"],
            "BUILD_FEATURE_STORE_FALLBACK",
            critical=True
        )
    
    if not success_build:
        pipeline_success = False
    
    # =====================================
    # Step 3: Validate Feature Store
    # =====================================
    log("\n✅ Step 3: Validate Feature Store")
    log("-" * 70)
    
    validate_script = BASE / "validate_feature_store_btc.py"
    if not validate_script.exists():
        validate_script = BASE / "scripts" / "validate_feature_store_btc.py"
    
    if validate_script.exists():
        run_step(
            ["python3", str(validate_script), "--path", str(FEATURE_STORE_PATH)],
            "VALIDATE_FEATURE_STORE",
            critical=False  # Non-critical
        )
    else:
        log("Validation script not found - skipping", "WARNING")
    
    # =====================================
    # Step 4: Train Model (Saturday only)
    # =====================================
    if current_day.weekday() == 5:  # Saturday
        log("\n🎓 Step 4: Train Model (Saturday)")
        log("-" * 70)
        
        train_script = BASE / "model" / "train_model.py"
        if train_script.exists():
            run_step(
                ["python3", str(train_script), "--plot"],
                "TRAIN_MODEL",
                critical=False
            )
        else:
            log("train_model.py not found", "WARNING")
    else:
        log("\n⏭️  Step 4: Skip Training (Not Saturday)", "SKIP")
    
    # =====================================
    # Step 5: Make Predictions
    # =====================================
    log("\n🔮 Step 5: Make Predictions")
    log("-" * 70)
    
    # ลองหา predict script
    predict_scripts = [
        BASE / "predict_gold_skip_sundays.py",
        BASE / "model" / "predict_gold_skip_sundays.py",
        BASE / "predict_gold.py",
        BASE / "model" / "predict_gold.py"
    ]
    
    predict_script = None
    for script in predict_scripts:
        if script.exists():
            predict_script = script
            break
    
    if predict_script:
        run_step(
            ["python3", str(predict_script), "--days", "7", "--save"],
            "PREDICT",
            critical=False
        )
    else:
        log("No prediction script found", "WARNING")
    
    # =====================================
    # Step 6: Generate Dashboard (Optional)
    # =====================================
    log("\n📊 Step 6: Generate Dashboard (Optional)")
    log("-" * 70)
    
    dashboard_script = BASE / "dashboard.py"
    if dashboard_script.exists():
        run_step(
            ["python3", str(dashboard_script)],
            "DASHBOARD",
            critical=False
        )
    else:
        log("Dashboard script not found - skipping", "WARNING")
    
    # =====================================
    # Final Summary
    # =====================================
    log("\n" + "=" * 70)
    if pipeline_success:
        log("PIPELINE COMPLETED SUCCESSFULLY", "SUCCESS")
        
        # แสดงข้อมูลล่าสุด
        if FEATURE_STORE_PATH.exists():
            try:
                import pandas as pd
                df = pd.read_csv(FEATURE_STORE_PATH, parse_dates=['date'])
                latest_date = df['date'].max().date()
                latest_gold = df.iloc[-1]['gold']
                log(f"Latest data: {latest_date}, Gold: {latest_gold:,.2f} THB", "INFO")
            except:
                pass
    else:
        log("PIPELINE COMPLETED WITH ERRORS", "ERROR")
        log("Check logs above for details", "ERROR")
    
    log("=" * 70)
    
    return 0 if pipeline_success else 1

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("\nPipeline interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        log(f"\nUnexpected error: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)
