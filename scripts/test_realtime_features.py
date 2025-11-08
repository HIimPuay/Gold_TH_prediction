# scripts/test_realtime_features_btc.py
from pathlib import Path
import pandas as pd
from realtime_features_btc import (
    load_context, 
    make_realtime_row, 
    build_feature_vector_for_today
)

# กำหนดที่อยู่ไฟล์ feature store
path = Path("../data/Feature_store/feature_store.csv")

# โหลด context 14 วันล่าสุด
context_df = load_context(path, context_days=14)

# ข้อมูลวันนี้ (ต้องมีครบทั้ง 6 ตัว)
payload = {
    "gold": 39000,      # ราคาทองคำ
    "fx": 32.1,         # อัตราแลกเปลี่ยน
    "cpi": 100.8,       # ดัชนีราคาผู้บริโภค
    "oil": 85.4,        # ราคาน้ำมัน
    "set": 1388.5,      # ดัชนีตลาดหลักทรัพย์
    "btc": 105000       # <<<< เพิ่ม Bitcoin >>>>
}

# สร้างแถวข้อมูลวันนี้
today_df = make_realtime_row("2025-11-08", payload)

# สร้างฟีเจอร์สำหรับการทำนาย
X_today = build_feature_vector_for_today(context_df, today_df)

print("✅ ฟีเจอร์ของวันนี้:")
print(X_today)
print("\n📊 จำนวนฟีเจอร์:", len(X_today.columns))
print("📋 รายชื่อฟีเจอร์:")
print(X_today.columns.tolist())