#!/usr/bin/env python3
"""
health_check.py - ตรวจสอบสุขภาพของระบบ

ใช้ก่อนรัน pipeline เพื่อดูว่าระบบพร้อมหรือไม่
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW_DIR = BASE / "data" / "raw"
FEATURE_STORE = BASE / "data" / "Feature_store" / "feature_store.csv"
MODEL_PATH = BASE / "model" / "best_model.pkl"

class HealthChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def check(self, name, func):
        """รันการตรวจสอบหนึ่งรายการ"""
        try:
            result = func()
            if result is True:
                self.passed.append(name)
                print(f"✅ {name}")
                return True
            elif result is False:
                self.issues.append(name)
                print(f"❌ {name}")
                return False
            else:  # Warning
                self.warnings.append(name)
                print(f"⚠️  {name}: {result}")
                return None
        except Exception as e:
            self.issues.append(f"{name}: {e}")
            print(f"❌ {name}: {e}")
            return False
    
    def print_summary(self):
        """แสดงสรุปผล"""
        print("\n" + "=" * 60)
        print("📊 HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"✅ Passed:   {len(self.passed)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Issues:   {len(self.issues)}")
        
        if self.issues:
            print("\n🔴 Critical Issues:")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print("\n🟡 Warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print("=" * 60)
        
        if len(self.issues) == 0:
            print("✅ System is healthy and ready!")
            return 0
        else:
            print("❌ System has issues. Please fix before running pipeline.")
            print("\n💡 Run: python3 fix_data_issues.py")
            return 1

def check_dependencies():
    """ตรวจสอบ dependencies"""
    try:
        import pandas
        import numpy
        import sklearn
        import joblib
        import requests
        return True
    except ImportError as e:
        return f"Missing dependency: {e.name}"

def check_raw_gold():
    """ตรวจสอบข้อมูลทอง"""
    gold_file = RAW_DIR / "gold_history.csv"
    if not gold_file.exists():
        return False
    
    df = pd.read_csv(gold_file)
    if len(df) == 0:
        return "Empty file"
    
    # ตรวจสอบคอลัมน์
    if 'datetime' not in df.columns and 'date' not in df.columns:
        return "Missing date column"
    
    # ตรวจสอบราคา
    price_cols = ['gold_sell', 'gold_bar_sell', 'gold_buy', 'gold_bar_buy']
    has_price = any(col in df.columns for col in price_cols)
    if not has_price:
        return "Missing price columns"
    
    return True

def check_raw_fx():
    """ตรวจสอบข้อมูล USD/THB"""
    fx_file = RAW_DIR / "exchange_rate.csv"
    if not fx_file.exists():
        return False
    
    df = pd.read_csv(fx_file)
    if len(df) == 0:
        return "Empty file"
    
    return True

def check_raw_cpi():
    """ตรวจสอบข้อมูล CPI"""
    cpi_file = RAW_DIR / "CPI_clean_for_supabase.csv"
    if not cpi_file.exists():
        return False
    
    df = pd.read_csv(cpi_file)
    if len(df) == 0:
        return "Empty file"
    
    return True

def check_raw_oil():
    """ตรวจสอบข้อมูลน้ำมัน"""
    oil_file = RAW_DIR / "petroleum_data.csv"
    if not oil_file.exists():
        return False
    
    df = pd.read_csv(oil_file)
    if len(df) == 0:
        return "Empty file"
    
    return True

def check_raw_set():
    """ตรวจสอบข้อมูล SET"""
    set_file = RAW_DIR / "set_index.csv"
    if not set_file.exists():
        return False
    
    df = pd.read_csv(set_file)
    if len(df) == 0:
        return "Empty file"
    
    return True

def check_raw_btc():
    """ตรวจสอบข้อมูล Bitcoin (optional)"""
    btc_file = RAW_DIR / "bitcoin_history.csv"
    if not btc_file.exists():
        return "Not found (optional)"
    
    df = pd.read_csv(btc_file)
    if len(df) == 0:
        return "Empty file"
    
    return True

def check_feature_store():
    """ตรวจสอบ feature store"""
    if not FEATURE_STORE.exists():
        return "Not found (will be created)"
    
    df = pd.read_csv(FEATURE_STORE, parse_dates=['date'])
    
    if len(df) == 0:
        return "Empty file"
    
    # เช็คคอลัมน์ที่จำเป็น
    required = ['date', 'gold', 'fx', 'cpi', 'oil', 'set', 'gold_next']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"Missing columns: {missing}"
    
    # เช็คว่าข้อมูลเก่าไหม
    latest = df['date'].max()
    days_old = (datetime.now() - latest).days
    
    if days_old > 7:
        return f"Data is {days_old} days old"
    elif days_old > 3:
        return f"Data is {days_old} days old (consider updating)"
    
    return True

def check_model():
    """ตรวจสอบโมเดล"""
    if not MODEL_PATH.exists():
        return "Model not trained yet"
    
    try:
        import joblib
        metadata_path = MODEL_PATH.parent / "model_metadata.pkl"
        if not metadata_path.exists():
            return "Missing metadata"
        
        metadata = joblib.load(metadata_path)
        
        # เช็คว่าโมเดลเก่าไหม
        trained_at = datetime.fromisoformat(metadata['trained_at'])
        days_old = (datetime.now() - trained_at).days
        
        if days_old > 30:
            return f"Model is {days_old} days old (consider retraining)"
        
        return True
        
    except Exception as e:
        return f"Error loading model: {e}"

def check_disk_space():
    """ตรวจสอบพื้นที่ disk"""
    import shutil
    
    total, used, free = shutil.disk_usage(BASE)
    free_gb = free // (2**30)
    
    if free_gb < 1:
        return f"Low disk space: {free_gb}GB free"
    elif free_gb < 5:
        return f"Disk space getting low: {free_gb}GB free"
    
    return True

def check_api_connectivity():
    """ตรวจสอบการเชื่อมต่อ API"""
    try:
        import requests
        
        # ทดสอบ Gold API
        response = requests.get(
            "https://api.chnwt.dev/thai-gold-api/latest",
            timeout=5
        )
        if response.status_code != 200:
            return f"Gold API returned {response.status_code}"
        
        # ทดสอบ CoinGecko API
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=thb",
            timeout=5
        )
        if response.status_code != 200:
            return "CoinGecko API issue (Bitcoin data may fail)"
        
        return True
        
    except requests.RequestException as e:
        return f"Network issue: {e}"

def main():
    print("=" * 60)
    print("🏥 SYSTEM HEALTH CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base: {BASE}\n")
    
    checker = HealthChecker()
    
    # Core checks
    print("📦 Dependencies:")
    checker.check("Python packages", check_dependencies)
    
    print("\n📁 Raw Data Files:")
    checker.check("Gold prices", check_raw_gold)
    checker.check("Exchange rates", check_raw_fx)
    checker.check("CPI data", check_raw_cpi)
    checker.check("Oil prices", check_raw_oil)
    checker.check("SET index", check_raw_set)
    checker.check("Bitcoin prices", check_raw_btc)
    
    print("\n🗄️  Processed Data:")
    checker.check("Feature store", check_feature_store)
    
    print("\n🤖 Model:")
    checker.check("Trained model", check_model)
    
    print("\n🌐 External Services:")
    checker.check("API connectivity", check_api_connectivity)
    
    print("\n💾 System Resources:")
    checker.check("Disk space", check_disk_space)
    
    return checker.print_summary()

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Health check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)