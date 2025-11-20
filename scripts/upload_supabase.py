# scripts/upload_supabase.py
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import numpy as np
import os

TABLE = "feature_store"
CSV_PATH = Path("data/Feature_store/feature_store.csv")

def to_py(v):
    """ทำให้ค่าเป็นชนิดที่ JSON serialize ได้"""
    if pd.isna(v):
        return None
    # pandas/NumPy → Python
    if isinstance(v, (np.generic,)):
        return v.item()
    # Timestamp/Datetime/Date → str
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    if hasattr(v, "isoformat"):  # เช่น datetime.date
        return v.isoformat()
    return v

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        raise SystemExit("CSV missing 'date' column")

    # ให้คอลัมน์ date เป็นสตริง YYYY-MM-DD (สำคัญมาก)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # ตัดคอลัมน์ไร้ชื่อที่อาจติดมา
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df

def rows_to_upsert_payload(df: pd.DataFrame):
    """โครงสร้างแบบเก็บ payload JSON (date + payload)"""
    cols = [c for c in df.columns if c != "date"]
    for _, row in df.iterrows():
        payload = {c: to_py(row[c]) for c in cols}
        yield {"date": to_py(row["date"]), "payload": payload}

def chunker(seq, size=500):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def main():
    load_dotenv()
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE"]
    sb = create_client(url, key)

    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = prepare_df(df)

    rows = list(rows_to_upsert_payload(df))  # ใช้แบบ payload
    total = 0
    for batch in chunker(rows, 500):
        sb.table(TABLE).upsert(batch, on_conflict="date").execute()
        total += len(batch)

    print(f"[{datetime.now()}] Upserted to '{TABLE}': {total} rows")

if __name__ == "__main__":
    main()
