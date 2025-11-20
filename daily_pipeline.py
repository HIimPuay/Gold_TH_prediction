import subprocess as sp
from datetime import datetime, timezone, timedelta
from pathlib import Path

ICT = timezone(timedelta(hours=7))
BASE = Path(__file__).resolve().parent

def run(cmd, name):
    print(f"[{datetime.now(ICT)}] >>> {name}")
    ret = sp.run(cmd, cwd=BASE)
    if ret.returncode != 0:
        print(f"⚠️  WARNING: {name} failed with code {ret.returncode}")
        print(f"   Continuing pipeline...")
        return False
    return True

def main():
    print(f"[{datetime.now(ICT)}] === DAILY PIPELINE START ===")

    # 1. ดึงข้อมูลทอง + Bitcoin พร้อมกัน (API)
    success_data = run(["python3", "scripts/ingest_gold.py"], "INGEST_DATA")
    
    # 3. รวมข้อมูลรายวัน
    run(["python3", "scripts/data_alignment_steps_btc.py"], "ALIGN_DATA")

    # 4. สร้าง Feature Store
    run(["python3", "scripts/build_feature_store_btc.py"], "BUILD_FEATURE_STORE")

    # 5. ตรวจสอบ Feature Store
    run(["python3", "scripts/validate_feature_store_btc.py", 
         "--path", "data/Feature_store/feature_store.csv"], "VALIDATE_FEATURE_STORE")
    
    # 6. Train model ใหม่ (ทุกวันอาทิตย์)
    if datetime.now(ICT).weekday() == 6:  # Sunday
        print("\n📅 Sunday - Retraining model...")
        run(["python3", "model/train_model.py", "--plot"], "TRAIN_MODEL")
    
    # 7. ทำนายราคา 7 วันข้างหน้า
    print("\n🔮 Making predictions...")
    run(["python3", "model/predict_gold.py", "--days", "7", "--save"], "PREDICT")

    print(f"[{datetime.now(ICT)}] === PIPELINE COMPLETED ===")
    
    # Summary
    if success_data:
        print("\n✅ Data updated successfully")
else:
    print("\n⚠️  Data update failed (check logs)")

if __name__ == "__main__":
    main()