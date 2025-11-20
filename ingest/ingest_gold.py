#!/usr/bin/env python3
"""
ingest_gold.py - ดึงข้อมูลทองจาก API (+ Bitcoin)
Version: 2.0 - Production Ready
"""
import os
import re
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
GOLD_FILE = RAW_DIR / "gold_history.csv"
BTC_FILE = RAW_DIR / "bitcoin_history.csv"

# APIs
GOLD_API = "https://api.chnwt.dev/thai-gold-api/latest"
BTC_API = "https://api.coingecko.com/api/v3/simple/price"

# Thai month mapping
THAI_MONTHS = {
    "มกราคม": "01", "กุมภาพันธ์": "02", "มีนาคม": "03", "เมษายน": "04",
    "พฤษภาคม": "05", "มิถุนายน": "06", "กรกฎาคม": "07", "สิงหาคม": "08",
    "กันยายน": "09", "ตุลาคม": "10", "พฤศจิกายน": "11", "ธันวาคม": "12"
}

# ==================== HELPERS ====================
def normalize_date(date_str):
    """แปลงวันที่จาก API ให้เป็น dd/mm/yyyy (พ.ศ.)"""
    if date_str is None:
        return None

    date_str = str(date_str).strip()

    # กรณี: 11/11/2568
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", date_str):
        return date_str

    # กรณี: 11 พฤศจิกายน 2568
    parts = date_str.split()
    if len(parts) == 3:
        day = parts[0]
        month_th = parts[1]
        year = parts[2]

        if month_th in THAI_MONTHS:
            month = THAI_MONTHS[month_th]
            return f"{int(day):02d}/{month}/{year}"

    return date_str

def normalize_time(time_str):
    """แปลงเวลาให้เป็น HH:MM:SS"""
    if time_str is None:
        return None

    time_str = str(time_str).strip()

    # ถ้ามีรูปแบบ 16:28
    m = re.search(r"(\d{1,2}:\d{2})", time_str)
    if m:
        time_hm = m.group(1)
        # เติม :00 ให้เป็น HH:MM:SS
        if len(time_hm.split(":")) == 2:
            return time_hm + ":00"
        return time_hm

    return None

def convert_buddhist_to_gregorian(date_str):
    """แปลง พ.ศ. เป็น ค.ศ."""
    if not date_str:
        return None
    
    try:
        parts = date_str.split("/")
        if len(parts) == 3:
            day, month, year = parts
            year = int(year)
            if year > 2400:  # ถ้าเป็น พ.ศ.
                year -= 543
            return f"{year:04d}-{int(month):02d}-{int(day):02d}"
    except:
        pass
    
    return None

# ==================== FETCHERS ====================
def fetch_gold_api():
    """ดึงข้อมูลราคาทองจาก API"""
    try:
        resp = requests.get(GOLD_API, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        response = data.get("response", {})
        price = response.get("price", {})

        gold = price.get("gold", {})
        gold_bar = price.get("gold_bar", {})

        # Normalize date/time
        normalized_date = normalize_date(response.get("date"))
        normalized_time = normalize_time(response.get("update_time"))
        
        # สร้าง datetime string
        gregorian_date = convert_buddhist_to_gregorian(normalized_date)
        if gregorian_date and normalized_time:
            datetime_str = f"{gregorian_date} {normalized_time}"
        else:
            datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "datetime": datetime_str,
            "date": normalized_date,
            "update_time": normalized_time,
            "gold_buy": gold.get("buy"),
            "gold_sell": gold.get("sell"),
            "gold_bar_buy": gold_bar.get("buy"),
            "gold_bar_sell": gold_bar.get("sell"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_url": GOLD_API
        }

        df = pd.DataFrame([row])

        # Clean numeric columns
        for col in ["gold_buy", "gold_sell", "gold_bar_buy", "gold_bar_sell"]:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

        print(f"✅ Gold: {df.iloc[0]['gold_sell']:,.2f} THB at {normalized_date} {normalized_time}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ Gold API error: {e}")
        return None
    except Exception as e:
        print(f"❌ Gold parsing error: {e}")
        return None

def fetch_btc_api():
    """ดึงข้อมูล Bitcoin (BTC/THB)"""
    try:
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'thb',
            'include_24hr_change': 'true'
        }
        
        resp = requests.get(BTC_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        btc_price = data['bitcoin']['thb']
        change_24h = data['bitcoin'].get('thb_24h_change', 0)
        
        row = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'btc_price': btc_price,
            'change_24h': change_24h,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.DataFrame([row])
        print(f"✅ Bitcoin: {btc_price:,.2f} THB ({change_24h:+.2f}%)")
        return df
        
    except Exception as e:
        print(f"❌ Bitcoin API error: {e}")
        return None

# ==================== SAVE & MERGE ====================
def save_gold(df_new):
    """บันทึกข้อมูลทอง (append + deduplicate)"""
    if df_new is None or df_new.empty:
        print("⚠️  No gold data to save")
        return False
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if GOLD_FILE.exists():
        df_old = pd.read_csv(GOLD_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # ลบข้อมูลซ้ำ (ใช้ datetime เป็นหลัก)
    if "datetime" in df_all.columns:
        df_all = df_all.drop_duplicates(subset=["datetime"], keep="last")
    else:
        df_all = df_all.drop_duplicates(subset=["date", "update_time"], keep="last")
    
    # Sort by datetime
    if "datetime" in df_all.columns:
        df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
        df_all = df_all.dropna(subset=["datetime"])
        df_all = df_all.sort_values("datetime")

    df_all.to_csv(GOLD_FILE, index=False, encoding="utf-8-sig", float_format="%.2f")
    print(f"💾 Gold saved: {GOLD_FILE} ({len(df_all)} records)")
    return True

def save_btc(df_new):
    """บันทึกข้อมูล Bitcoin (append + deduplicate)"""
    if df_new is None or df_new.empty:
        print("⚠️  No Bitcoin data to save")
        return False
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if BTC_FILE.exists():
        df_old = pd.read_csv(BTC_FILE, parse_dates=['date'])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # Convert date และลบซ้ำ
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.sort_values('date')
    df_all = df_all.drop_duplicates(subset=['date'], keep='last')

    df_all.to_csv(BTC_FILE, index=False, encoding="utf-8-sig", float_format="%.2f")
    print(f"💾 Bitcoin saved: {BTC_FILE} ({len(df_all)} records)")
    return True

# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("🚀 GOLD & BITCOIN DATA INGESTION")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Fetch Gold
    print("1️⃣  Fetching Gold prices...")
    df_gold = fetch_gold_api()
    gold_success = save_gold(df_gold)

    print()

    # Fetch Bitcoin
    print("2️⃣  Fetching Bitcoin prices...")
    df_btc = fetch_btc_api()
    btc_success = save_btc(df_btc)

    print()
    print("=" * 70)
    
    if gold_success and btc_success:
        print("✅ ALL DATA UPDATED SUCCESSFULLY")
    elif gold_success:
        print("⚠️  PARTIAL SUCCESS (Gold only)")
    elif btc_success:
        print("⚠️  PARTIAL SUCCESS (Bitcoin only)")
    else:
        print("❌ FAILED TO UPDATE ANY DATA")
    
    print("=" * 70)

if __name__ == "__main__":
    main()