# test_supabase.py
from supabase import create_client
from dotenv import load_dotenv
import os
import json

# โหลดไฟล์ .env แบบระบุ path ตรง ๆ (กัน error AssertionError)
load_dotenv(dotenv_path="/Users/nichanun/Desktop/DSDN/.env")

# ดึงค่าจาก environment
url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_SERVICE_ROLE"]

# เชื่อมต่อ Supabase
sb = create_client(url, key)

print("🚀 Connecting to Supabase...")

# ดึงข้อมูล 3 แถวล่าสุดจากตาราง feature_store
res = sb.table("feature_store").select("*").order("date", desc=True).limit(3).execute()

print("✅ Connection successful!")
print(f"Found {len(res.data)} rows.\n")

# แสดงข้อมูลในรูปอ่านง่าย
if res.data:
    for row in res.data:
        print("📅 Date:", row["date"])
        print("📦 Payload:")
        print(json.dumps(row["payload"], indent=2, ensure_ascii=False))
        print("-" * 50)
else:
    print("⚠️  ไม่มีข้อมูลในตาราง feature_store ครับ")
