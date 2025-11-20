# 🚨 แก้ปัญหาด่วน - Feature Names Mismatch

## ปัญหาที่พบ:

```
Missing features: ['gold_roll7_mean', 'gold_pct_change', ...]
```

**สาเหตุ:** โมเดลถูก train ด้วย feature names แบบหนึ่ง แต่ feature store สร้างชื่ออีกแบบ

## 🔧 วิธีแก้ (3 ขั้นตอน):

### 1. รันสคริปต์แก้ไขด่วน

```bash
cd /Users/nichanun/Desktop/DSDN
python3 fix_urgent_issues.py
```

สคริปต์นี้จะ:
- ✅ แก้ชื่อคอลัมน์ใน feature_store.csv
- ✅ สร้างไฟล์ USD_THB_Historical Data.csv
- ✅ แก้ Bitcoin data columns
- ✅ สร้าง data_alignment_fixed.py
- ✅ แก้ daily_pipeline.py

### 2. Build feature store ใหม่

```bash
# ใช้สคริปต์แก้ไขแล้ว
python3 build_feature_store_fixed.py --btc data/raw/bitcoin_history.csv
```

หรือถ้าต้องการใช้ข้อมูลที่มีอยู่:

```bash
# ใช้ data_alignment ที่แก้แล้ว
python3 scripts/data_alignment_fixed.py

# สร้าง feature store
python3 build_feature_store_fixed.py --btc data/raw/bitcoin_history.csv
```

### 3. Train model ใหม่

```bash
# Train ด้วยข้อมูลใหม่
python3 model/train_model.py --plot

# ทดสอบทำนาย
python3 model/predict_gold.py --days 7 --save
```

---

## 📋 วิธีแก้แบบละเอียด

### ปัญหาที่ 1: Feature names ไม่ตรงกัน

**โมเดลต้องการ:**
- `gold_roll7_mean`, `gold_pct_change`

**Feature store สร้าง:**
- `gold_roll7`, `gold_pct`

**วิธีแก้:**

Option A: แก้ชื่อใน feature store (เร็วกว่า)
```bash
python3 fix_urgent_issues.py
```

Option B: Train model ใหม่ด้วย features ที่มี
```bash
# Rebuild feature store ด้วย script ใหม่
python3 build_feature_store_fixed.py --btc data/raw/bitcoin_history.csv

# Train ใหม่
python3 model/train_model.py --plot
```

### ปัญหาที่ 2: ไฟล์ USD_THB_Historical Data.csv หายไป

**วิธีแก้:**

```python
# สร้างจาก exchange_rate.csv
import pandas as pd

df = pd.read_csv('data/raw/exchange_rate.csv')
df['Date'] = pd.to_datetime(df['period'].astype(str) + '-01')
df['Price'] = df['mid_rate']
df[['Date', 'Price']].to_csv('data/raw/USD_THB_Historical Data.csv', index=False)
```

หรือใช้ `data_alignment_fixed.py` ที่จะใช้ `exchange_rate.csv` โดยตรง

### ปัญหาที่ 3: Bitcoin data columns

**วิธีแก้:**

```python
import pandas as pd

df = pd.read_csv('data/raw/bitcoin_history.csv')

# แก้ชื่อคอลัมน์
if 'Date' in df.columns:
    df = df.rename(columns={'Date': 'date'})
if 'Close' in df.columns:
    df = df.rename(columns={'Close': 'btc_price'})

df[['date', 'btc_price']].to_csv('data/raw/bitcoin_history.csv', index=False)
```

---

## ✅ Checklist หลังแก้ไข

- [ ] Feature store มีคอลัมน์ `gold_roll7_mean`, `gold_pct_change`
- [ ] ไฟล์ `USD_THB_Historical Data.csv` มีอยู่ หรือใช้ `data_alignment_fixed.py`
- [ ] Bitcoin data มีคอลัมน์ `date` และ `btc_price`
- [ ] โมเดล train สำเร็จ
- [ ] การทำนายทำงานได้

ตรวจสอบ:
```bash
# เช็ค feature store
python3 -c "
import pandas as pd
df = pd.read_csv('data/Feature_store/feature_store.csv')
print('Columns:', df.columns.tolist())
print('Has roll7_mean:', 'gold_roll7_mean' in df.columns)
print('Has pct_change:', 'gold_pct_change' in df.columns)
"

# เช็คการทำนาย
python3 model/predict_gold.py --days 1
```

---

## 🎯 Quick Fix ที่สุด

ถ้าอยากแก้ไขเร็วที่สุด:

```bash
# 1. แก้ทุกอย่างในครั้งเดียว
python3 fix_urgent_issues.py

# 2. Build feature store ใหม่ (ใช้ชื่อที่ถูกต้อง)
python3 build_feature_store_fixed.py --btc data/raw/bitcoin_history.csv

# 3. Train model ใหม่
python3 model/train_model.py

# 4. ทดสอบ
python3 model/predict_gold.py --days 7 --save
```

เสร็จแค่นี้! 🎉

---

## 💡 Tips เพิ่มเติม

1. **เก็บ backup** - `fix_urgent_issues.py` จะ backup อัตโนมัติ

2. **เช็คก่อนรัน**:
   ```bash
   python3 health_check.py
   ```

3. **ถ้ายังไม่ได้** - rebuild ทั้งหมด:
   ```bash
   python3 ingest/ingest_gold.py
   python3 scripts/data_alignment_fixed.py
   python3 build_feature_store_fixed.py --btc data/raw/bitcoin_history.csv
   python3 model/train_model.py --plot
   python3 model/predict_gold.py --days 7 --save
   ```

---

*Last updated: 2025-11-19*
