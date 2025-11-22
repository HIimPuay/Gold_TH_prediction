# ✅ Installation Checklist

## 📋 ก่อนเริ่มติดตั้ง

- [ ] อยู่ที่ directory: `/Users/nichanun/Desktop/DSDN`
- [ ] มีโฟลเดอร์ `data/` อยู่
- [ ] มีไฟล์ `data/raw/gold_history.csv` อยู่
- [ ] Python 3.8+ installed

---

## 🔧 ขั้นตอนการติดตั้ง

### 1. Backup ไฟล์เก่า
```bash
cd /Users/nichanun/Desktop/DSDN
mkdir -p backup_$(date +%Y%m%d)
cp daily_pipeline.py backup_$(date +%Y%m%d)/ 2>/dev/null
cp build_feature_store_btc.py backup_$(date +%Y%m%d)/ 2>/dev/null
```
- [ ] Backup เสร็จแล้ว

### 2. คัดลอกไฟล์ใหม่
```bash
# คัดลอกไฟล์ทั้งหมดจาก download มายัง /Users/nichanun/Desktop/DSDN
```
- [ ] คัดลอก `gold_config.py`
- [ ] คัดลอก `predict_gold_skip_sundays.py`
- [ ] คัดลอก `daily_pipeline.py`

### 3. ตรวจสอบไฟล์
```bash
ls -la gold_config.py
ls -la predict_gold_skip_sundays.py
ls -la daily_pipeline.py
```
- [ ] ไฟล์ทั้ง 3 อยู่ใน directory ถูกต้อง

### 4. ตั้งค่าระบบ
```bash
# แก้ไข gold_config.py
nano gold_config.py
# หรือ
open -e gold_config.py
```
- [ ] เปลี่ยน `GOLD_PRICE_TYPE = "gold_bar_sell"`
- [ ] บันทึกไฟล์

---

## 🧪 ทดสอบระบบ

### Test 1: Syntax Check
```bash
python3 -m py_compile predict_gold_skip_sundays.py
python3 -m py_compile daily_pipeline.py
```
- [ ] ไม่มี syntax error

### Test 2: Config Load
```bash
python3 -c "from gold_config import GOLD_PRICE_TYPE; print(GOLD_PRICE_TYPE)"
```
- [ ] แสดง `gold_bar_sell` (หรือค่าที่ตั้งไว้)

### Test 3: Build Feature Store
```bash
python3 build_feature_store_btc.py
```
- [ ] แสดง `[INFO] Using gold price type: gold_bar_sell`
- [ ] แสดง `[OK] Feature store saved`

### Test 4: Prediction
```bash
python3 predict_gold_skip_sundays.py --days 1
```
- [ ] แสดง `📊 Using: gold_bar_sell`
- [ ] แสดงการทำนาย
- [ ] ไม่มี error

### Test 5: Pipeline (ถ้าไม่ใช่วันอาทิตย์)
```bash
python3 daily_pipeline.py
```
- [ ] ถ้าเป็นวันอาทิตย์: แสดง "PIPELINE SKIPPED (SUNDAY)"
- [ ] ถ้าไม่ใช่: รันได้ครบทุก step

---

## ✅ ตรวจสอบผลลัพธ์

### Feature Store
```bash
tail -3 data/Feature_store/feature_store.csv
```
- [ ] มีข้อมูลล่าสุด
- [ ] คอลัมน์ `gold` มีค่า

### Predictions
```bash
ls -lt results/predictions_*.csv | head -1
cat $(ls -t results/predictions_*.csv | head -1)
```
- [ ] มีไฟล์การทำนาย
- [ ] แสดง `price_type: gold_bar_sell`

---

## 🚀 Setup Automation (Optional)

### ตั้งค่า Cron
```bash
crontab -e
```

เพิ่มบรรทัด:
```
0 17 * * * cd /Users/nichanun/Desktop/DSDN && /bin/zsh run_daily.sh
```
- [ ] ตั้งค่า cron แล้ว
- [ ] ทดสอบรัน manual ได้

---

## 🎯 การใช้งานประจำวัน

### ทุกวัน (ยกเว้นอาทิตย์)
```bash
cd /Users/nichanun/Desktop/DSDN
python3 daily_pipeline.py
```
- [ ] รู้วิธีรัน pipeline

### ทำนายราคา
```bash
# 1 วัน
python3 predict_gold_skip_sundays.py --days 1 --save

# 7 วัน
python3 predict_gold_skip_sundays.py --days 7 --save
```
- [ ] รู้วิธีทำนาย

### ดูผลลัพธ์
```bash
# Feature store
tail data/Feature_store/feature_store.csv

# Predictions
cat results/predictions_*.csv
```
- [ ] รู้วิธีดูผลลัพธ์

---

## 📞 Troubleshooting

### ปัญหา: ImportError gold_config
```bash
# ตรวจสอบ path
pwd  # ต้องได้ /Users/nichanun/Desktop/DSDN
ls gold_config.py  # ต้องมีไฟล์นี้
```
- [ ] แก้ไขแล้ว

### ปัญหา: Syntax Error
```bash
# ดาวน์โหลดไฟล์ใหม่อีกครั้ง (มี bug fix แล้ว)
```
- [ ] แก้ไขแล้ว

### ปัญหา: Feature store ไม่อัพเดท
```bash
# รัน manual
python3 ingest_gold.py
python3 build_feature_store_btc.py
```
- [ ] แก้ไขแล้ว

---

## 📚 เอกสารที่ควรอ่าน

- [ ] `README.md` - ภาพรวมทั้งหมด
- [ ] `QUICK_START.md` - คู่มือเริ่มต้น 5 นาที
- [ ] `BUGFIX_NOTES.md` - รายละเอียด bug ที่แก้ไข
- [ ] `SYSTEM_UPDATE_SUMMARY.md` - เอกสารฉบับเต็ม

---

## ✨ Bonus Tips

### ดูว่าใช้ราคาประเภทไหน
```bash
# วิธีที่ 1
cat gold_config.py | grep GOLD_PRICE_TYPE

# วิธีที่ 2
python3 predict_gold_skip_sundays.py --days 1 | head -5
```

### Backup อัตโนมัติ
```bash
# เพิ่มใน crontab (ทุกวันเสาร์ 23:00)
0 23 * * 6 cd /Users/nichanun/Desktop/DSDN && tar -czf backup_$(date +\%Y\%m\%d).tar.gz data/ model/ results/
```

### Monitor Accuracy
```bash
python3 -c "
import joblib
m = joblib.load('model/model_metadata.pkl')
print(f\"MAE: {m['metrics']['MAE']:.2f} THB\")
print(f\"RMSE: {m['metrics']['RMSE']:.2f} THB\")
print(f\"R²: {m['metrics']['R2']:.4f}\")
"
```

---

## 🎉 เสร็จสมบูรณ์!

เมื่อทำทุกขั้นตอนเสร็จแล้ว:
- [ ] ระบบรันได้ปกติ
- [ ] ทราบว่าใช้ราคาทองประเภทไหน (gold_bar_sell)
- [ ] ข้ามวันอาทิตย์อัตโนมัติ
- [ ] สามารถทำนายราคาได้
- [ ] เข้าใจวิธีใช้งานพื้นฐาน

**ยินดีด้วย! ระบบพร้อมใช้งานแล้ว 🚀**

---

**Last Updated:** 23 November 2025  
**Version:** 2.0  
**Status:** Ready for Production ✅
