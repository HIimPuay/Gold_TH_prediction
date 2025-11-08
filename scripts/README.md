# Thai Gold Price Feature Store
ระบบสร้าง Feature Store สำหรับคาดการณ์ราคาทองคำไทย 96.5%

## โครงงานนี้ทำอะไร
- รวม Raw data หลายแหล่ง (ทอง, USD/THB, CPI, น้ำมัน, SET)
- Align ความถี่เป็นรายวัน
- จัดการ missing values
- สร้าง **Target** (`gold_next`), **Lag (1, 3 วัน)**, **Rolling (7 วัน)**, และ **pct_change**
- ส่งออกไฟล์ `feature_store.csv` พร้อมนำเข้า Supabase / ใช้ฝั่ง DS ได้ทันที

## โครงสร้างไฟล์อินพุต (วางไว้ในโฟลเดอร์เดียวกัน)
- `gold_history.csv` – มีคอลัมน์ `date` (รูปแบบไทย `dd/mm/yyyy` พ.ศ.) และราคาทอง เช่น `gold_sell`
- `exchange_rate.csv` – BOT: มี `period` (YYYY-MM) และคอลัมน์อัตราแลกเปลี่ยน เช่น `mid_rate`
- `CPI_clean_for_supabase.csv` – มี `code`, `date`, `cpi_index` (ใช้ `code = 0` = CPI รวม)
- `petroleum_data.csv` – EIA: มี `period` (YYYY-MM) และ `value` (ถ้ามีหลายซีรีส์จะ **เฉลี่ย** รายเดือน)
- `set_index.csv` – มี `date`, `Close` (ล้าง header แปลก ๆ ออกโดยสคริปต์)

> หากคอลัมน์ชื่อไม่ตรง สคริปต์จะพยายามเลือกคอลัมน์ตัวเลขที่เหมาะสมอัตโนมัติ

## การใช้งาน
```bash
# ติดตั้งไลบรารีที่จำเป็น
pip install pandas numpy

# รันสคริปต์ (อยู่โฟลเดอร์เดียวกับไฟล์ .csv)
python build_feature_store.py \
  --data-dir . \
  --out ./feature_store.csv \
  --roll-window 7 \
  --min-periods 3
```

## สคีมาของ Feature Store (คีย์หลัก = `date`)
- `date` (PK, รายวัน)
- สัญญาณดิบ: `gold`, `fx`, `cpi`, `oil`, `set`
- Target: `gold_next` (ราคาทองวันถัดไป)
- Lags: `{var}_lag1`, `{var}_lag3`
- Rolling mean: `{var}_roll7_mean`
- เปอร์เซ็นต์การเปลี่ยนแปลง: `{var}_pct_change`

> `{var}` = `gold|fx|cpi|oil|set`

## ตัวอย่าง DDL (PostgreSQL/Supabase)
```sql
CREATE TABLE IF NOT EXISTS feature_store (
  date            date PRIMARY KEY,
  gold            numeric,
  fx              numeric,
  cpi             numeric,
  oil             numeric,
  set             numeric,

  gold_next       numeric,

  gold_lag1       numeric,
  gold_lag3       numeric,
  fx_lag1         numeric,
  fx_lag3         numeric,
  cpi_lag1        numeric,
  cpi_lag3        numeric,
  oil_lag1        numeric,
  oil_lag3        numeric,
  set_lag1        numeric,
  set_lag3        numeric,

  gold_roll7_mean numeric,
  fx_roll7_mean   numeric,
  cpi_roll7_mean  numeric,
  oil_roll7_mean  numeric,
  set_roll7_mean  numeric,

  gold_pct_change numeric,
  fx_pct_change   numeric,
  cpi_pct_change  numeric,
  oil_pct_change  numeric,
  set_pct_change  numeric
);
```

## ขั้นตอนที่สคริปต์ทำ
1. แปลงวันที่:
   - ทอง (`dd/mm/yyyy` พ.ศ.) → ค.ศ.
   - FX/Oil monthly (`YYYY-MM`) → ต้นเดือน (`YYYY-MM-01`)
2. รวมเข้าปฏิทินรายวัน (business days)
3. Align ความถี่:
   - `cpi`/`fx`/`oil` → **forward-fill** รายวัน
   - `gold`/`set` → **forward-fill + backfill** เพื่ออุดวันหยุดตลาด
4. จัดการ Missing values ที่เหลือด้วย `interpolate`
5. สร้าง `gold_next` (shift -1), lags (1,3), rolling 7 วัน, `pct_change`
6. Drop แถว NaN ที่เกิดจากการเลื่อน/rolling แล้วเซฟ CSV

## เคล็ดลับ Production
- เลือกซีรีส์น้ำมัน (Brent/WTI) ให้ชัดเจนแทนการเฉลี่ย หากต้องการความแม่นยำขึ้น
- เพิ่ม data validation (เช็ค outlier, ช่วงวันที่ต่อเนื่อง)
- ทำ logging/report หลังรัน (จำนวนแถว, ช่วงวันที่, ค่า NaN)
- แยกเวอร์ชัน Feature Store (`fs_version`) เมื่อปรับพารามิเตอร์

## ใบอนุญาต
สำหรับงานวิชา/วิจัยการศึกษา
