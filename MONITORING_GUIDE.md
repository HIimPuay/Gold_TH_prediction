# 📊 Model Monitoring & Performance Tracking

## 🎯 Overview

ระบบนี้ติดตาม performance ของโมเดลแบบ real-time และตรวจจับ **Concept Drift** เพื่อแจ้งเตือนเมื่อควรทำ retrain

---

## 📈 Metrics ที่ติดตาม

### 1. **Moving Average Accuracy (MAcc)**

**คำนวณ:**
```
MAcc(t, w) = (1/w) × Σ Acc(t-k)  for k=0 to w-1
```

**อธิบาย:**
- ค่าเฉลี่ย accuracy ใน window (เช่น 7 วัน, 30 วัน)
- ช่วยให้เห็น trend ระยะยาว
- ลด noise จากการแกว่งรายวัน

**Window sizes:**
- **7 วัน** → แนวโน้มระยะสั้น (sensitive)
- **30 วัน** → แนวโน้มระยะยาว (stable)

**ตัวอย่าง:**
```
Date       | Accuracy | MAcc_7d | MAcc_30d
-----------|----------|---------|----------
2025-11-13 | 98.5%    | 98.2%   | 97.8%
2025-11-14 | 97.8%    | 98.0%   | 97.8%
2025-11-15 | 96.9%    | 97.7%   | 97.7%
...
```

---

### 2. **Accuracy Decay Rate**

**คำนวณ:**
```
DecayRate = (Acc(t1) - Acc(t2)) / (t2 - t1)
```

**อธิบาย:**
- วัดความเร็วที่ accuracy กำลังลดลง
- หน่วย: % per day
- ใช้ตรวจจับ drift ก่อนจะเกิดปัญหา

**เกณฑ์การแจ้งเตือน:**
- DecayRate < -0.5% → ⚠️ Warning
- DecayRate < -1.0% → 🚨 Critical

**ตัวอย่าง:**
```
DecayRate = (98.0% - 97.0%) / 1 day = -1.0% per day
→ 🚨 Accuracy กำลังลดลงเร็ว! ควร investigate
```

---

### 3. **Concept Drift Detection**

**ตรวจจับด้วย 2 วิธี:**

**A. Decay Rate Threshold:**
```
if DecayRate < -0.5%:
    drift_detected = True
```

**B. Consecutive Decline:**
```
if Accuracy ลดลงต่อเนื่อง 3+ วัน:
    drift_detected = True
```

**Severity Levels:**
- **LOW:** DecayRate > -0.5%
- **MEDIUM:** -1.0% < DecayRate < -0.5%
- **HIGH:** DecayRate < -1.0%

**Actions:**
| Severity | Action |
|----------|--------|
| LOW | Monitor only |
| MEDIUM | Investigate features, check data quality |
| HIGH | **Retrain model immediately** |

---

### 4. **Balance Index (BI)**

**คำนวณ:**
```
BI = (จำนวนที่ predict > actual - จำนวนที่ predict < actual) / total
```

**ช่วงค่า:** -1 ถึง +1

**ตีความ:**
```
BI > +0.1  → Over-predict  (ทำนายสูงกว่าจริง)
BI < -0.1  → Under-predict (ทำนายต่ำกว่าจริง)
-0.1 ≤ BI ≤ +0.1 → Balanced
```

**ตัวอย่าง:**
```
Total predictions: 100
Over-predict: 65 ครั้ง
Under-predict: 35 ครั้ง
BI = (65 - 35) / 100 = +0.30 → โมเดลมีแนวโน้ม over-predict
```

**การแก้:**
- Retrain with more balanced data
- ปรับ model penalty/weight

---

### 5. **Adaptation Speed**

**คำนวณจาก:**
- **Variance** ของ accuracy (10 วันล่าสุด)
- **Trend** (slope ของ accuracy)

**Classification:**
```
Variance < 0.1  → STABLE   (โมเดลเสถียร)
Variance < 0.5  → MODERATE (แกว่งปานกลาง)
Variance ≥ 0.5  → VOLATILE (แกว่งมาก)
```

**Trend:**
```
slope > 0  → IMPROVING (accuracy กำลังดีขึ้น)
slope < 0  → DECLINING (accuracy กำลังแย่ลง)
slope ≈ 0  → STABLE    (คงที่)
```

**ตัวอย่าง:**
```
Speed: MODERATE
Trend: IMPROVING
→ โมเดลกำลังปรับตัวและดีขึ้น ✅
```

---

## 🚀 วิธีใช้งาน

### 1. ติดตั้ง

```bash
cd /Users/nichanun/Desktop/DSDN

# ดาวน์โหลด model_monitoring.py ไปที่ project
cp ~/Downloads/model_monitoring.py .
```

### 2. เตรียมข้อมูล Predictions

สร้างไฟล์ `predictions_history.csv`:
```csv
date,actual,predicted
2025-11-01,61400,61350
2025-11-02,61450,61480
2025-11-03,61500,61520
...
```

### 3. รัน Monitoring

```python
import pandas as pd
from model_monitoring import generate_monitoring_report, print_monitoring_report

# โหลดข้อมูล
predictions_df = pd.read_csv('predictions_history.csv', parse_dates=['date'])

# Generate report
report, macc_df = generate_monitoring_report(predictions_df)

# แสดงผล
print_monitoring_report(report)
```

### 4. ผลลัพธ์ที่ได้

```
======================================================================
📊 MODEL PERFORMANCE MONITORING REPORT
======================================================================
📅 Generated: 2025-11-20 15:10:23
📈 Data Points: 50
🗓️  Date Range: 2025-10-01 to 2025-11-19

🎯 ACCURACY METRICS
----------------------------------------------------------------------
Current (7-day):   97.85%
Current (30-day):  98.12%
Decay Rate (7d):   -0.234% per day
Decay Rate (30d):  -0.089% per day

🔍 CONCEPT DRIFT DETECTION
----------------------------------------------------------------------
7-day window:   ✅ NO DRIFT (Severity: LOW)
30-day window:  ✅ NO DRIFT (Severity: LOW)

⚖️  BALANCE INDEX
----------------------------------------------------------------------
Value: +0.078
Interpretation: BALANCED

🏃 ADAPTATION METRICS
----------------------------------------------------------------------
Speed:    STABLE
Trend:    IMPROVING
Variance: 0.0823

💡 RECOMMENDATIONS
----------------------------------------------------------------------
   ✅ Model performance is stable.

======================================================================
```

---

## 🎯 การใช้ใน Production

### 1. รัน Monitoring ทุกวัน

เพิ่มใน `daily_pipeline.py`:

```python
from model_monitoring import generate_monitoring_report, print_monitoring_report, save_monitoring_history

# หลังจาก predict
predictions_df = pd.read_csv('results/predictions_7days_latest.csv')
report, macc_df = generate_monitoring_report(predictions_df)

# Alert ถ้าเจอ drift
if report['drift_detection']['7d']['drift_detected']:
    print("🚨 ALERT: Concept drift detected!")
    # ส่ง email/notification
    
# Save history
save_monitoring_history(report)
```

### 2. ตั้ง Threshold สำหรับ Auto-retrain

```python
if report['accuracy']['current_7d'] < 95.0:
    print("📉 Accuracy below threshold. Starting retrain...")
    os.system("python3 model/train_model.py")
```

### 3. Dashboard Integration

```python
# ใน dashboard.py
def show_monitoring_dashboard():
    report, macc_df = generate_monitoring_report(predictions_df)
    
    print("\n🔍 Model Health:")
    print(f"   7-day Accuracy: {report['accuracy']['current_7d']:.2f}%")
    print(f"   Drift Status: {'🚨 DETECTED' if report['drift_detection']['7d']['drift_detected'] else '✅ OK'}")
    print(f"   Balance: {report['balance_index']['interpretation']}")
```

---

## 📊 ตัวอย่าง Use Cases

### Case 1: Drift Detection

```
Day 1: MAcc_7d = 98.5%
Day 2: MAcc_7d = 98.3%
Day 3: MAcc_7d = 98.0%
Day 4: MAcc_7d = 97.5%
Day 5: MAcc_7d = 96.8%

DecayRate = (98.5 - 96.8) / 4 = -0.425% per day
→ ⚠️ Warning! Approaching drift threshold
```

**Action:** Investigate data quality, check for market changes

### Case 2: Balanced Model

```
Total predictions: 100
Over-predict: 52
Under-predict: 48
BI = (52-48)/100 = +0.04

→ ✅ Model is well-balanced
```

### Case 3: Volatile Performance

```
Last 10 days accuracy: [98, 95, 99, 94, 98, 93, 99, 94, 98, 95]
Variance = 5.28

→ 🎢 VOLATILE - Monitor closely
```

---

## ✅ Best Practices

1. **รัน monitoring ทุกวัน** หลังจาก prediction
2. **เก็บ history อย่างน้อย 30 วัน** เพื่อดู trend
3. **ตั้ง alert** เมื่อ DecayRate < -0.5%
4. **Retrain ทันที** เมื่อ drift severity = HIGH
5. **เช็ค Balance Index** หลัง retrain เสมอ

---

## 📁 ไฟล์ที่เกี่ยวข้อง

- `model_monitoring.py` - ระบบ monitoring หลัก
- `results/model_monitoring.json` - ประวัติ monitoring
- `results/moving_avg_accuracy_*.csv` - ข้อมูล MAcc
- `daily_pipeline.py` - เรียกใช้ monitoring อัตโนมัติ

---

*Last updated: 2025-11-20*
