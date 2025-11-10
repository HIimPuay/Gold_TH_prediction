import os
import requests
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
API_URL = "https://api.chnwt.dev/thai-gold-api/latest"
RAW_FILE = "data/raw/ingest_gold.csv"
# ----------------------------

def fetch_and_save_gold_data():
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)

    resp = requests.get(API_URL)
    resp.raise_for_status()
    data = resp.json()

    response = data.get("response", {})
    price = response.get("price", {})

    gold = price.get("gold", {})
    gold_bar = price.get("gold_bar", {})
    change = price.get("change", {})

    # รวมข้อมูลทั้งหมดใน 1 แถว
    row = {
        "date": response.get("date"),
        "update_time": response.get("update_time"),
        "gold_buy": gold.get("buy"),
        "gold_sell": gold.get("sell"),
        "gold_bar_buy": gold_bar.get("buy"),
        "gold_bar_sell": gold_bar.get("sell"),
        "change_prev": change.get("compare_previous"),
        "change_yesterday": change.get("compare_yesterday"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df_new = pd.DataFrame([row])
    

    # ถ้ามีไฟล์อยู่แล้ว → อ่านของเก่ามาต่อท้าย
    if os.path.exists(RAW_FILE):
        df_old = pd.read_csv(RAW_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # เขียนทับไฟล์เดิม (แต่ข้อมูลถูก append แล้ว)
    df_all.to_csv(RAW_FILE, index=False, encoding="utf-8-sig")

    print(f"✅ Updated data saved to {RAW_FILE}")
    print(df_new)

if __name__ == "__main__":
    fetch_and_save_gold_data()
