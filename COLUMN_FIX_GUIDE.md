# 🔧 แก้ปัญหาชื่อคอลัมน์ใน Feature Store

## 🐛 ปัญหาที่พบ

โมเดล ML ที่เทรนไว้คาดหวังชื่อคอลัมน์แบบหนึ่ง แต่ feature store ที่สร้างมีชื่อคอลัมน์อีกแบบหนึ่ง:

### ชื่อที่โมเดลต้องการ:
```
gold_roll7_mean    ← Rolling mean 7 วัน
gold_pct_change    ← Percentage change
fx_roll7_mean
fx_pct_change
...
```

### ชื่อที่ feature store สร้างมา:
```
gold_roll7    ❌ ไม่มี _mean
gold_pct      ❌ ไม่มี _change
fx_roll7      ❌ 
fx_pct        ❌
...
```

### ผลกระทบ:
- โมเดลทำนายไม่ได้ เพราะหาคอลัมน์ไม่เจอ
- Error: `KeyError: 'gold_roll7_mean'`

---

## ✅ วิธีแก้ไข (3 วิธี)

### วิธีที่ 1: ใช้สคริปต์แก้ไขอัตโนมัติ ⭐ แนะนำ

```bash
cd /Users/nichanun/Desktop/DSDN

# แก้ไข feature store ที่มีอยู่
python3 fix_feature_store_columns.py --backup

# ผลลัพธ์:
# ✅ Backup created: feature_store_backup_20251123.csv
# ✅ Fixed: 10 columns renamed
```

**สิ่งที่สคริปต์ทำ:**
1. อ่าน `feature_store.csv`
2. สร้าง backup อัตโนมัติ
3. เปลี่ยนชื่อคอลัมน์:
   - `gold_roll7` → `gold_roll7_mean`
   - `gold_pct` → `gold_pct_change`
   - (และตัวแปรอื่น ๆ ทั้งหมด)
4. บันทึกกลับ

### วิธีที่ 2: Build ใหม่ด้วย Script ที่แก้ไขแล้ว

```bash
# ใช้ build_feature_store_fixed.py จากเอกสาร 33
cd /Users/nichanun/Desktop/DSDN

# Build feature store ใหม่
python3 build_feature_store_fixed.py

# ผลลัพธ์:
# ✅ Feature store saved with correct column names
```

### วิธีที่ 3: แก้ด้วย Python โดยตรง

```python
import pandas as pd

# อ่านไฟล์
df = pd.read_csv('data/Feature_store/feature_store.csv')

# เปลี่ยนชื่อคอลัมน์
rename_map = {}
for var in ['gold', 'fx', 'cpi', 'oil', 'set', 'btc']:
    rename_map[f'{var}_roll7'] = f'{var}_roll7_mean'
    rename_map[f'{var}_pct'] = f'{var}_pct_change'

df = df.rename(columns=rename_map)

# บันทึก
df.to_csv('data/Feature_store/feature_store.csv', index=False)
print("✅ Fixed!")
```

---

## 🔍 ตรวจสอบว่าแก้แล้วหรือยัง

### เช็คชื่อคอลัมน์
```bash
head -1 data/Feature_store/feature_store.csv | tr ',' '\n' | grep -E "roll|pct"
```

**ต้องเห็น:**
```
gold_roll7_mean     ✅
gold_pct_change     ✅
fx_roll7_mean       ✅
fx_pct_change       ✅
cpi_roll7_mean      ✅
cpi_pct_change      ✅
...
```

**ไม่ควรเห็น:**
```
gold_roll7    ❌
gold_pct      ❌
```

### ทดสอบการทำนาย
```bash
python3 predict_gold_skip_sundays.py --days 1
```

**ถ้าแก้แล้ว:**
```
✅ Loaded model
✅ Loaded data
🔮 Predicting...
💎 Predicted price: 42,150.00 บาท
```

**ถ้ายังไม่แก้:**
```
❌ Error: Missing features: ['gold_roll7_mean', 'gold_pct_change', ...]
```

---

## 📋 Checklist การแก้ไข

### ก่อนแก้ไข
- [ ] Backup feature store เดิม
- [ ] ตรวจสอบว่ามี `fix_feature_store_columns.py`
- [ ] อยู่ที่ directory ถูกต้อง (`/Users/nichanun/Desktop/DSDN`)

### แก้ไข
- [ ] รันคำสั่ง: `python3 fix_feature_store_columns.py --backup`
- [ ] เห็นข้อความ "✅ Feature store fixed successfully!"

### หลังแก้ไข
- [ ] เช็คชื่อคอลัมน์ว่าถูกต้อง
- [ ] ทดสอบทำนาย: `python3 predict_gold_skip_sundays.py --days 1`
- [ ] ควรทำนายได้โดยไม่มี error

---

## 🎯 วิธีป้องกันปัญหานี้ในอนาคต

### 1. ใช้ build_feature_store_fixed.py เสมอ
```bash
# แทนที่จะใช้
python3 build_feature_store_btc.py

# ใช้
python3 build_feature_store_fixed.py
```

### 2. อัพเดท daily_pipeline.py
แก้ไขในไฟล์ `daily_pipeline.py`:

```python
# เดิม
run(["python3", "build_feature_store_btc.py"], ...)

# ใหม่
run(["python3", "build_feature_store_fixed.py"], ...)
```

### 3. เพิ่ม Validation
เพิ่มในสคริปต์:
```python
# ตรวจสอบชื่อคอลัมน์
required_cols = [
    'gold_roll7_mean',  # ไม่ใช่ gold_roll7
    'gold_pct_change',  # ไม่ใช่ gold_pct
    'fx_roll7_mean',
    'fx_pct_change',
    # ...
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")
```

---

## 🚨 Troubleshooting

### ปัญหา: หลังแก้แล้วยังทำนายไม่ได้
**สาเหตุ:**
1. โมเดลเก่าเทรนด้วยชื่อคอลัมน์ผิด
2. Feature store ยังไม่ rebuild

**วิธีแก้:**
```bash
# 1. แก้ feature store
python3 fix_feature_store_columns.py --backup

# 2. เทรนโมเดลใหม่
python3 model/train_model.py

# 3. ทดสอบทำนาย
python3 predict_gold_skip_sundays.py --days 1
```

### ปัญหา: ไฟล์ backup เยอะเกินไป
```bash
# ลบ backup เก่า (เก็บแค่ 5 ไฟล์ล่าสุด)
cd data/Feature_store
ls -t feature_store_backup_*.csv | tail -n +6 | xargs rm -f
```

### ปัญหา: ต้องการกลับไปใช้ backup
```bash
# ดู backup ที่มี
ls -lt data/Feature_store/feature_store_backup_*.csv

# กลับไปใช้ backup
cp data/Feature_store/feature_store_backup_20251123_103045.csv \
   data/Feature_store/feature_store.csv
```

---

## 📊 สรุปการเปลี่ยนแปลง

| ชื่อเดิม | ชื่อใหม่ | ตัวแปร |
|----------|----------|--------|
| `gold_roll7` | `gold_roll7_mean` ✅ | Rolling mean 7 วัน |
| `gold_pct` | `gold_pct_change` ✅ | Percentage change |
| `fx_roll7` | `fx_roll7_mean` ✅ | Rolling mean 7 วัน |
| `fx_pct` | `fx_pct_change` ✅ | Percentage change |
| `cpi_roll7` | `cpi_roll7_mean` ✅ | Rolling mean 7 วัน |
| `cpi_pct` | `cpi_pct_change` ✅ | Percentage change |
| `oil_roll7` | `oil_roll7_mean` ✅ | Rolling mean 7 วัน |
| `oil_pct` | `oil_pct_change` ✅ | Percentage change |
| `set_roll7` | `set_roll7_mean` ✅ | Rolling mean 7 วัน |
| `set_pct` | `set_pct_change` ✅ | Percentage change |

*(ถ้ามี Bitcoin: `btc_roll7`, `btc_pct` ก็เปลี่ยนเหมือนกัน)*

---

## 💡 หมายเหตุ

### ทำไมต้องเปลี่ยนชื่อ?
- โมเดลที่เทรนไว้ใช้ชื่อคอลัมน์แบบเต็ม (`_mean`, `_change`)
- Feature engineering best practice ใช้ชื่อที่บ่งบอกความหมายชัดเจน
- ป้องกันความสับสนระหว่าง features ต่าง ๆ

### ต้องเทรนโมเดลใหม่หรือไม่?
**ไม่จำเป็น** ถ้า:
- แค่แก้ชื่อคอลัมน์ใน feature store
- โมเดลเก่ายังใช้ได้

**ควรเทรนใหม่** ถ้า:
- Build feature store ใหม่ทั้งหมด
- มีข้อมูลใหม่เพิ่มเข้ามาเยอะ
- ต้องการปรับปรุง accuracy

---

**Last Updated:** 23 November 2025  
**Issue:** Column naming mismatch  
**Status:** ✅ Fixed with `fix_feature_store_columns.py`
