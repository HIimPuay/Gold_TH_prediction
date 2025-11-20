#!/usr/bin/env python3
"""
ingest_bitcoin.py - ดึงข้อมูล Bitcoin (BTC/THB) จาก API
"""
import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path("/Users/nichanun/Desktop/DSDN/data/raw")
BTC_FILE = BASE_DIR / "bitcoin_history.csv"

def fetch_btc_current():
    """ดึงราคา BTC ล่าสุดจาก CoinGecko API (Free, No API Key)"""
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        'ids': 'bitcoin',
        'vs_currencies': 'thb',
        'include_24hr_change': 'true'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        btc_price = data['bitcoin']['thb']
        change_24h = data['bitcoin']['thb_24h_change']
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'btc_price': btc_price,
            'change_24h': change_24h,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ Error fetching BTC: {e}")
        return None

def fetch_btc_historical(days=730):
    """
    ดึงข้อมูล BTC ย้อนหลัง (สำหรับครั้งแรก)
    days=730 → ประมาณ 2 ปี
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        'vs_currency': 'thb',
        'days': days,
        'interval': 'daily'
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # แปลงเป็น DataFrame
        prices = data['prices']  # [[timestamp_ms, price], ...]
        df = pd.DataFrame(prices, columns=['timestamp', 'btc_price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
        df = df.groupby('date').agg({'btc_price': 'last'}).reset_index()
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"✅ Fetched {len(df)} days of BTC history")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching historical BTC: {e}")
        return None

def load_btc_history():
    """โหลดประวัติ BTC ที่มีอยู่"""
    if BTC_FILE.exists():
        df = pd.read_csv(BTC_FILE, parse_dates=['date'])
        return df
    return pd.DataFrame(columns=['date', 'btc_price'])

def merge_btc_data(df_existing, df_new):
    """รวมข้อมูล BTC เก่า+ใหม่ และลบซ้ำ"""
    df = pd.concat([df_existing, df_new], ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.drop_duplicates(subset=['date'], keep='last')
    return df

def main():
    print("🪙 Bitcoin Data Ingestion")
    print("=" * 60)
    
    # ตรวจสอบว่ามีข้อมูลเก่าหรือไม่
    df_existing = load_btc_history()
    
    if df_existing.empty:
        print("📥 No existing data, fetching historical...")
        df_new = fetch_btc_historical(days=730)
    else:
        print(f"📁 Found {len(df_existing)} existing records")
        print(f"   Latest: {df_existing['date'].max().date()}")
        print("📥 Fetching latest price...")
        
        # ดึงราคาล่าสุด
        latest = fetch_btc_current()
        if latest:
            df_new = pd.DataFrame([latest])
            df_new['date'] = pd.to_datetime(df_new['date'])
        else:
            print("❌ Failed to fetch latest data")
            return
    
    if df_new is not None and not df_new.empty:
        # รวมข้อมูล
        df_final = merge_btc_data(df_existing, df_new)
        
        # บันทึก
        BASE_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(BTC_FILE, index=False)
        
        print(f"\n✅ Saved {len(df_final)} records to: {BTC_FILE}")
        print(f"   Date range: {df_final['date'].min().date()} to {df_final['date'].max().date()}")
        print(f"   Latest BTC: {df_final.iloc[-1]['btc_price']:,.2f} THB")
    else:
        print("❌ No data to save")

if __name__ == "__main__":
    main()