#!/usr/bin/env python3
"""
prepare_gold_data.py - เตรียมข้อมูลทองจากไฟล์ที่ user upload

ข้อมูลมี 874 แถว ตั้งแต่ 02/01/2566 ถึง 31/10/2568
ราคาแปรปรวนจาก ~29,750 ถึง ~67,200 บาท
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import re

# Paths
UPLOADED_FILE = Path("/mnt/user-data/uploads/gold_history.csv")
OUTPUT_DIR = Path("/mnt/user-data/outputs")
TARGET_FILE = Path("/Users/nichanun/Desktop/DSDN/data/raw/gold_history.csv")

print("=" * 70)
print("🔧 เตรียมข้อมูลทอง (Real Data)")
print("=" * 70)

# ==================== LOAD DATA ====================
print("\n1️⃣  โหลดข้อมูล...")

df = pd.read_csv(UPLOADED_FILE)
print(f"   📊 Loaded: {len(df)} rows")
print(f"   📋 Columns: {df.columns.tolist()}")

# ==================== CLEAN DATA ====================
print("\n2️⃣  ทำความสะอาดข้อมูล...")

# แปลงวันที่จาก พ.ศ. เป็น ค.ศ.
def convert_thai_date(date_str):
    """แปลง dd/mm/yyyy (พ.ศ.) เป็น datetime (ค.ศ.)"""
    if pd.isna(date_str):
        return pd.NaT
    
    try:
        date_str = str(date_str).strip()
        
        # รูปแบบ dd/mm/yyyy (พ.ศ.)
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                day, month, year = parts
                year = int(year)
                
                # แปลง พ.ศ. → ค.ศ.
                if year > 2400:
                    year -= 543
                
                return pd.to_datetime(f"{year:04d}-{int(month):02d}-{int(day):02d}")
    except:
        pass
    
    return pd.NaT

# แปลงวันที่
df['datetime'] = df['date'].apply(convert_thai_date)
df = df.dropna(subset=['datetime'])

print(f"   ✅ Converted dates: {len(df)} rows")
print(f"   📅 Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")

# ทำความสะอาดราคา
price_columns = ['gold_buy', 'gold_sell', 'gold_bar_buy', 'gold_bar_sell']
for col in price_columns:
    if col in df.columns:
        # แปลงเป็น numeric (ลบ comma ถ้ามี)
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ลบข้อมูลที่ invalid
df = df.dropna(subset=['gold_sell'])
df = df[df['gold_sell'] > 0]

print(f"   ✅ Cleaned prices: {len(df)} valid rows")

# ==================== STATISTICS ====================
print("\n3️⃣  สถิติข้อมูล...")

print(f"   📊 Total rows: {len(df)}")
print(f"   📅 Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
print(f"   💰 Gold sell price:")
print(f"      Min:    {df['gold_sell'].min():,.2f} THB")
print(f"      Max:    {df['gold_sell'].max():,.2f} THB")
print(f"      Mean:   {df['gold_sell'].mean():,.2f} THB")
print(f"      Median: {df['gold_sell'].median():,.2f} THB")
print(f"      Std:    {df['gold_sell'].std():,.2f} THB")
print(f"   📈 Unique prices: {df['gold_sell'].nunique()}")
print(f"   📊 Daily change (avg): {df['gold_sell'].diff().mean():,.2f} THB")

# ==================== PREPARE OUTPUT ====================
print("\n4️⃣  เตรียมไฟล์สำหรับใช้งาน...")

# เรียงตามวันที่
df = df.sort_values('datetime')

# สร้างคอลัมน์ที่จำเป็น
df_output = pd.DataFrame({
    'datetime': df['datetime'],
    'date': df['date'],  # เก็บวันที่ไทยไว้ด้วย
    'update_time': df['update_time'],
    'gold_buy': df['gold_buy'],
    'gold_sell': df['gold_sell'],
    'gold_bar_buy': df['gold_bar_buy'],
    'gold_bar_sell': df['gold_bar_sell'],
    'timestamp': df['timestamp'],
    'source_url': df['source_url']
})

# ลบข้อมูลซ้ำ (ถ้ามี)
df_output = df_output.drop_duplicates(subset=['datetime'], keep='last')

print(f"   ✅ Prepared: {len(df_output)} rows")

# ==================== SAVE ====================
print("\n5️⃣  บันทึกไฟล์...")

# Save to outputs (สำหรับ download)
output_path = OUTPUT_DIR / "gold_history_cleaned.csv"
df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"   💾 Saved to outputs: {output_path}")

# แสดงข้อมูลตัวอย่าง
print("\n📋 ตัวอย่างข้อมูล 10 แถวแรก:")
print(df_output[['datetime', 'gold_sell', 'gold_buy']].head(10).to_string(index=False))

print("\n📋 ตัวอย่างข้อมูล 10 แถวสุดท้าย:")
print(df_output[['datetime', 'gold_sell', 'gold_buy']].tail(10).to_string(index=False))

# ==================== INSTRUCTIONS ====================
print("\n" + "=" * 70)
print("✅ เตรียมข้อมูลเสร็จสิ้น!")
print("=" * 70)

print(f"""
📁 ไฟล์ที่สร้าง:
   {output_path}

🎯 ขั้นตอนถัดไป:

   ใน Mac ของคุณ:
   
   1. คัดลอกไฟล์นี้ไปแทนที่:
      cp {output_path} {TARGET_FILE}
   
   หรือ
   
   2. ดาวน์โหลดจาก outputs และวางที่:
      /Users/nichanun/Desktop/DSDN/data/raw/gold_history.csv
   
   3. รัน pipeline:
      cd /Users/nichanun/Desktop/DSDN
      python3 scripts/build_feature_store_btc.py
      python3 model/train_model.py --plot
      python3 model/predict_gold.py --days 7 --save

📊 ข้อมูลนี้:
   • มี {len(df_output)} วัน
   • ย้อนหลัง ~{(df_output['datetime'].max() - df_output['datetime'].min()).days} วัน
   • ราคาแปรปรวนจาก {df_output['gold_sell'].min():,.0f} ถึง {df_output['gold_sell'].max():,.0f} บาท
   • Std = {df_output['gold_sell'].std():,.2f} (ดีมาก!)
   
✅ ข้อมูลนี้เหมาะสำหรับ train model!
""")

# ==================== VALIDATION ====================
print("\n🔍 การตรวจสอบคุณภาพ...")

issues = []

# เช็คจำนวนแถว
if len(df_output) < 500:
    issues.append(f"⚠️  มีข้อมูลน้อย ({len(df_output)} แถว) แนะนำ > 500")
else:
    print(f"   ✅ จำนวนแถวเพียงพอ: {len(df_output)} แถว")

# เช็ค variance
if df_output['gold_sell'].std() < 1000:
    issues.append(f"⚠️  ความแปรปรวนต่ำ (Std={df_output['gold_sell'].std():.2f}) แนะนำ > 1000")
else:
    print(f"   ✅ ความแปรปรวนดี: Std = {df_output['gold_sell'].std():,.2f}")

# เช็ค missing values
if df_output['gold_sell'].isna().any():
    issues.append("⚠️  มี missing values ในราคาทอง")
else:
    print("   ✅ ไม่มี missing values")

# เช็ค duplicates
dup_count = df_output.duplicated(subset=['datetime']).sum()
if dup_count > 0:
    issues.append(f"⚠️  มีข้อมูลซ้ำ {dup_count} แถว")
else:
    print("   ✅ ไม่มีข้อมูลซ้ำ")

# เช็ค date continuity
date_gaps = (df_output['datetime'].diff().dt.days > 7).sum()
if date_gaps > 10:
    issues.append(f"⚠️  มีช่องว่างในข้อมูล > 7 วัน จำนวน {date_gaps} ครั้ง")
else:
    print(f"   ✅ ข้อมูลต่อเนื่อง (ช่องว่าง > 7 วัน: {date_gaps} ครั้ง)")

if issues:
    print("\n⚠️  ประเด็นที่ควรทราบ:")
    for issue in issues:
        print(f"   {issue}")
else:
    print("\n✅ ข้อมูลผ่านการตรวจสอบทั้งหมด!")

print("\n" + "=" * 70)
