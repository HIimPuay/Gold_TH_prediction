from pathlib import Path
import pandas as pd

# โฟลเดอร์ฐาน = โฟลเดอร์ data/
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"

src_csv = RAW_DIR / "CPI-G_Report.csv"   # ไฟล์ต้นฉบับ (ดิบ)
out_csv = RAW_DIR / "cpi_clean.csv"      # ไฟล์ผลลัพธ์ที่ต้องการ

df = pd.read_csv(src_csv, header=None)

# หา header จริง (แถวที่ 5)
df.columns = df.iloc[4]
df = df.drop(index=range(0, 5))
df = df.dropna(how="all")

# ตั้งชื่อคอลัมน์ต้นๆ
df = df.rename(columns={df.columns[0]: "code", df.columns[1]: "name_th", df.columns[2]: "year"})

# wide -> long
df_melt = df.melt(id_vars=["code", "name_th", "year"], var_name="month_th", value_name="cpi_index")

thai_month = {"ม.ค.":1,"ก.พ.":2,"มี.ค.":3,"เม.ย.":4,"พ.ค.":5,"มิ.ย.":6,"ก.ค.":7,"ส.ค.":8,"ก.ย.":9,"ต.ค.":10,"พ.ย.":11,"ธ.ค.":12}
df_melt["month"] = df_melt["month_th"].map(thai_month)

df_melt["date_month"] = pd.to_datetime(
    df_melt["year"].astype(int)*10000 + df_melt["month"].astype(int)*100 + 1,
    format="%Y%m%d"
)

df_total = df_melt[df_melt["code"] == "00000"][["date_month", "cpi_index"]].copy()
df_total["cpi_index"] = pd.to_numeric(df_total["cpi_index"], errors="coerce")

RAW_DIR.mkdir(parents=True, exist_ok=True)
df_total.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("✅ Saved:", out_csv)
