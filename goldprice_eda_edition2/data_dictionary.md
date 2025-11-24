# Data Dictionary – Feature Store

| Column | Dtype | Description |
|---|---|---|
| `date` | `object` | วันที่ (PK) ปฏิทินวันทำการ |
| `gold` | `float64` | ราคาทองไทย 96.5% ขายออก (THB ต่อบาททองคำ) |
| `fx` | `float64` | อัตราแลกเปลี่ยน USD/THB (บาทต่อดอลลาร์) |
| `cpi` | `float64` | ดัชนีราคาผู้บริโภครวม (index) |
| `oil` | `float64` | ตัวชี้วัดราคาน้ำมัน (เฉลี่ยรายเดือนจาก EIA; หน่วยตามแหล่งข้อมูล) |
| `set` | `float64` | SET Index (ดัชนีตลาดหุ้นไทย) |
| `gold_next` | `float64` | เป้าหมาย (ราคาทองวันถัดไป) |
| `gold_lag1` | `float64` | Lag 1 วันของ gold |
| `gold_lag3` | `float64` | Lag 3 วันของ gold |
| `gold_roll7_mean` | `float64` | ค่าเฉลี่ยเคลื่อนที่ 7 วันของ gol |
| `gold_pct_change` | `float64` | อัตราการเปลี่ยนแปลงรายวันของ gold (หน่วยเป็นสัดส่วน) |
| `fx_lag1` | `float64` | Lag 1 วันของ fx |
| `fx_lag3` | `float64` | Lag 3 วันของ fx |
| `fx_roll7_mean` | `float64` | ค่าเฉลี่ยเคลื่อนที่ 7 วันของ f |
| `fx_pct_change` | `float64` | อัตราการเปลี่ยนแปลงรายวันของ fx (หน่วยเป็นสัดส่วน) |
| `cpi_lag1` | `float64` | Lag 1 วันของ cpi |
| `cpi_lag3` | `float64` | Lag 3 วันของ cpi |
| `cpi_roll7_mean` | `float64` | ค่าเฉลี่ยเคลื่อนที่ 7 วันของ cp |
| `cpi_pct_change` | `float64` | อัตราการเปลี่ยนแปลงรายวันของ cpi (หน่วยเป็นสัดส่วน) |
| `oil_lag1` | `float64` | Lag 1 วันของ oil |
| `oil_lag3` | `float64` | Lag 3 วันของ oil |
| `oil_roll7_mean` | `float64` | ค่าเฉลี่ยเคลื่อนที่ 7 วันของ oi |
| `oil_pct_change` | `float64` | อัตราการเปลี่ยนแปลงรายวันของ oil (หน่วยเป็นสัดส่วน) |
| `set_lag1` | `float64` | Lag 1 วันของ set |
| `set_lag3` | `float64` | Lag 3 วันของ set |
| `set_roll7_mean` | `float64` | ค่าเฉลี่ยเคลื่อนที่ 7 วันของ se |
| `set_pct_change` | `float64` | อัตราการเปลี่ยนแปลงรายวันของ set (หน่วยเป็นสัดส่วน) |