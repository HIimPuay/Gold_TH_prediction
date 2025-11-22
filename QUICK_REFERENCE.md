# 🚀 Quick Reference Card - Gold Prediction System

## 📥 ติดตั้ง (5 นาที)

```bash
# 1. ดาวน์โหลดไฟล์ทั้งหมดมายัง
cd /Users/nichanun/Desktop/DSDN

# 2. แก้ปัญหาคอลัมน์ (สำคัญ!)
python3 fix_feature_store_columns.py --backup

# 3. ตั้งค่า
nano gold_config.py
# เปลี่ยน: GOLD_PRICE_TYPE = "gold_bar_sell"

# 4. ทดสอบ
python3 predict_gold_skip_sundays.py --days 1
```

---

## 🎯 คำสั่งที่ใช้บ่อย

### ทำนายราคา
```bash
# 1 วัน
python3 predict_gold_skip_sundays.py --days 1 --save

# 7 วัน
python3 predict_gold_skip_sundays.py --days 7 --save
```

### รัน Pipeline
```bash
python3 daily_pipeline.py
```

### ตรวจสอบ
```bash
# เช็คราคาที่ใช้
cat gold_config.py | grep GOLD_PRICE_TYPE

# เช็คชื่อคอลัมน์
head -1 data/Feature_store/feature_store.csv | tr ',' '\n' | grep roll

# ดูข้อมูลล่าสุด
tail -3 data/Feature_store/feature_store.csv
```

---

## 🚨 แก้ปัญหาด่วน

### Error: Missing features
```bash
python3 fix_feature_store_columns.py --backup
```

### Error: ImportError gold_config
```bash
pwd  # ต้องได้ /Users/nichanun/Desktop/DSDN
ls gold_config.py  # ต้องมีไฟล์นี้
```

### Error: SyntaxError
```bash
python3 -m py_compile predict_gold_skip_sundays.py
# ดาวน์โหลดไฟล์ใหม่ถ้ายัง error
```

---

## 📊 ตรวจสอบสุขภาพระบบ

```bash
# One-liner check all
python3 -c "
import pandas as pd
from pathlib import Path

# Check files
files = ['gold_config.py', 'data/Feature_store/feature_store.csv']
for f in files:
    print(f'{'✅' if Path(f).exists() else '❌'} {f}')

# Check feature store
df = pd.read_csv('data/Feature_store/feature_store.csv')
print(f'\n📊 Feature Store:')
print(f'   Rows: {len(df)}')
print(f'   Latest: {df.iloc[-1][\"date\"]}')
print(f'   Gold: {df.iloc[-1][\"gold\"]:,.2f} THB')

# Check columns
has_mean = 'gold_roll7_mean' in df.columns
has_change = 'gold_pct_change' in df.columns
print(f'\n{'✅' if has_mean and has_change else '❌'} Column names correct')
"
```

---

## 🎓 ประเภทราคาทอง

| Code | ประเภท | แนะนำ |
|------|--------|--------|
| `gold_bar_sell` | ทองแท่ง ราคาขาย | ⭐⭐⭐ |
| `gold_bar_buy` | ทองแท่ง ราคารับซื้อ | ⭐⭐ |
| `gold_sell` | ทองรูปพรรณ ราคาขาย | ⭐ |
| `gold_buy` | ทองรูปพรรณ ราคารับซื้อ | ⭐ |

---

## 📚 เอกสารอ้างอิง

| ไฟล์ | เนื้อหา | เวลา |
|------|---------|------|
| **START_HERE.md** | เริ่มต้น | 5 นาที |
| **COLUMN_FIX_GUIDE.md** | แก้คอลัมน์ | 5 นาที |
| **QUICK_START.md** | เริ่มใช้งาน | 5 นาที |
| **README.md** | ภาพรวม | 10 นาที |
| **SYSTEM_UPDATE_SUMMARY.md** | ทุกอย่าง | 30 นาที |

---

## ⚙️ Auto-run (Cron)

```bash
# เปิด crontab
crontab -e

# รันทุกวัน 17:00
0 17 * * * cd /Users/nichanun/Desktop/DSDN && python3 daily_pipeline.py
```

---

## 🔢 ผลลัพธ์ที่คาดหวัง

### ถูกต้อง ✅
```
📊 Using: gold_bar_sell
🔮 Predicting next 7 business days...
   (Skipping: Sunday)
Day 1: 2025-11-24 (Monday) 📈 42,150.00 บาท (+0.85%)
...
```

### ผิดพลาด ❌
```
❌ Error: Missing features: ['gold_roll7_mean', ...]
→ แก้: python3 fix_feature_store_columns.py --backup
```

---

## 💡 Tips

### สร้าง Alias
```bash
# เพิ่มใน ~/.zshrc
alias gpred="python3 /Users/nichanun/Desktop/DSDN/predict_gold_skip_sundays.py"
alias gpipe="python3 /Users/nichanun/Desktop/DSDN/daily_pipeline.py"

# ใช้งาน
gpred --days 7 --save
```

### Quick Status
```bash
# สร้างสคริปต์
cat > status.sh << 'EOF'
#!/bin/bash
echo "📊 System Status"
echo "Config: $(grep GOLD_PRICE_TYPE gold_config.py)"
echo "Latest: $(tail -1 data/Feature_store/feature_store.csv | cut -d',' -f1,2)"
echo "Columns: $(head -1 data/Feature_store/feature_store.csv | grep -o 'roll7_mean' | wc -l)/5 OK"
EOF

chmod +x status.sh
./status.sh
```

---

**Version:** 2.0  
**Updated:** 23 Nov 2025  
**Print & Keep!** 📌
