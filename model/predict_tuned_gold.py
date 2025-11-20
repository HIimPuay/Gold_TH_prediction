#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gold.py - ทำนายราคาทองคำวันถัดไป (รองรับทำนายหลายวันแบบ recursive)
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
import sys 

# ==================== PATH / CONFIG ==================== #

def find_project_root():
    """หา root directory ของโปรเจกต์"""
    current = Path.cwd()
    if current.name == "model":
        return current.parent
    if (current / "data" / "Feature_store").exists():
        return current
    if (current.parent / "data" / "Feature_store").exists():
        return current.parent
    return current

PROJECT_ROOT = find_project_root()
MODEL_DIR = PROJECT_ROOT / "model"
FEATURE_STORE = PROJECT_ROOT / "data" / "Feature_store" / "feature_store.csv"

# 🎯 NEW: กำหนดชื่อไฟล์โมเดลที่จูนแล้ว
TUNED_MODEL_FILENAME = "ridge_tuned.pkl" 

# ==================== CORE FUNCTIONS ==================== #

def load_model_and_metadata(model_dir: Path):
    """
    โหลดโมเดลและ metadata (ถูกแก้ไขให้โหลดโมเดลที่จูนแล้วเป็นหลัก)
    """
    
    # 1. NEW: ลองโหลดโมเดลที่จูนแล้วก่อน (ridge_tuned.pkl)
    model_path = model_dir / TUNED_MODEL_FILENAME
    
    # 2. Fallback: ถ้าไม่พบ ให้ใช้ best_model.pkl เดิม
    if not model_path.exists():
        model_path = model_dir / "best_model.pkl"
        
    metadata_path = model_dir / "model_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Cannot find any model at: {model_dir}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Metadata not found at: {metadata_path}")
    
    print(f"✅ Loading model from: {model_path.name}")
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    return model, metadata

def load_latest_data(path: Path):
    """โหลดข้อมูลล่าสุด"""
    if not path.exists():
        raise FileNotFoundError(f"❌ Feature store not found at: {path}")
    
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def _safe_ffill_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """เติมค่าให้ปลอดภัยเพื่อหลีกเลี่ยง NaN ในฟีเจอร์"""
    # ffill ก่อน (ใช้ค่าก่อนหน้า), ถ้ายัง NaN อยู่ให้เป็น 0
    return df.ffill().fillna(0)

def _parse_lag(col: str):
    """
    จับแพตเทิร์นชื่อคอลัมน์รูปแบบ base_lagN
    คืนค่า (base_name, N) หรือ (None, None) ถ้าไม่ใช่
    """
    m = re.match(r"^(.*)_lag(\d+)$", col)
    if not m:
        return None, None
    base = m.group(1)
    try:
        n = int(m.group(2))
    except Exception:
        n = None
    return base, n

def build_next_feature_row(last_row: pd.Series, feature_cols: list, predicted_price: float, next_date: pd.Timestamp) -> pd.Series:
    """
    สร้างแถวฟีเจอร์สำหรับ 'วันถัดไป' จากแถวล่าสุด + ราคาทองที่ทำนายได้
    """
    new_row = last_row.copy()

    # อัปเดตวัน
    new_row["date"] = pd.to_datetime(next_date)

    # อัปเดตราคาทอง (ถ้ามีคอลัมน์ gold)
    if "gold" in new_row.index:
        # prev_gold = float(last_row.get("gold", np.nan))
        new_row["gold"] = float(predicted_price)
    # else:
    # prev_gold = np.nan

    # เตรียม cache ของค่าฐาน เพื่อใช้อัปเดต lag
    base_value = {c: last_row.get(c, np.nan) for c in last_row.index}

    # อัปเดตคอลัมน์ *_lagN โดยพยายาม chain
    for col in last_row.index:
        base, n = _parse_lag(col)
        if base is None or n is None:
            continue

        # กรณีอัปเดต lag ของ gold ให้ใช้ predicted_price เป็นฐาน
        if base == "gold":
            if n == 1:
                new_row[col] = float(last_row.get("gold", np.nan))  # gold ของ "วันก่อนหน้า"
            else:
                prev_col = f"{base}_lag{n-1}"
                if prev_col in last_row.index:
                    new_row[col] = last_row.get(prev_col, np.nan)
                else:
                    new_row[col] = float(last_row.get("gold", np.nan))
        else:
            # สำหรับตัวแปรอื่น ๆ (usd_thb, set, oil ฯลฯ)
            if n == 1:
                new_row[col] = base_value.get(base, np.nan)
            else:
                prev_col = f"{base}_lag{n-1}"
                new_row[col] = base_value.get(prev_col, base_value.get(base, np.nan))

    # ทำความสะอาดค่า NaN เบื้องต้น (สำหรับคอลัมน์ที่ไม่ใช่ lag/predicted gold ให้คงค่าก่อนหน้าไว้)
    new_row = new_row.ffill().fillna(0)
    return new_row

def predict_next_day(model, df: pd.DataFrame, feature_cols: list):
    """ทำนายราคาทองวันถัดไป (ใช้แถวล่าสุดจาก df)"""
    # ใช้ข้อมูลแถวล่าสุด
    latest = df.iloc[-1:].copy()

    # ตรวจสอบว่ามีฟีเจอร์ครบหรือไม่
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # เตรียม features
    X = latest[feature_cols].copy()

    # ป้องกัน NaN
    if X.isna().any().any():
        print("⚠️  Warning: Found NaN in features, filling by ffill/0")
        X = _safe_ffill_zeros(X)

    # ทำนาย
    prediction = float(model.predict(X)[0])

    # ข้อมูลวันล่าสุด
    last_date = pd.to_datetime(df.iloc[-1]["date"])
    last_gold = float(df.iloc[-1]["gold"]) if "gold" in df.columns else np.nan

    # คำนวณการเปลี่ยนแปลง (ถ้ามี gold)
    if np.isnan(last_gold):
        change = np.nan
        change_pct = np.nan
    else:
        change = prediction - last_gold
        change_pct = (change / last_gold) * 100 if last_gold != 0 else np.nan

    return {
        "last_date": last_date,
        "last_price": last_gold,
        "predicted_price": prediction,
        "change": change,
        "change_pct": change_pct,
        "next_date": last_date + timedelta(days=1)
    }

def format_output(result: dict, metadata: dict):
    """จัดรูปแบบการแสดงผล"""
    print("\n" + "=" * 60)
    print("🔮 GOLD PRICE PREDICTION")
    print("=" * 60)
    print(f"\n📅 Last available date: {result['last_date'].strftime('%Y-%m-%d')}")
    if not np.isnan(result['last_price']):
        print(f"💰 Last gold price:     {result['last_price']:,.2f} บาท")
    else:
        print(f"💰 Last gold price:     -")
    print(f"\n📅 Prediction for:      {result['next_date'].strftime('%Y-%m-%d')}")
    print(f"💎 Predicted price:     {result['predicted_price']:,.2f} บาท")
    
    # แสดงการเปลี่ยนแปลง
    if not np.isnan(result['change']):
        change_symbol = "📈" if result['change'] > 0 else "📉" if result['change'] < 0 else "➡️"
        sign = "+" if result['change'] > 0 else ""
        pct_str = f"{sign}{result['change_pct']:.2f}%" if result['change_pct'] == result['change_pct'] else "-"
        print(f"\n{change_symbol} Change:             {sign}{result['change']:,.2f} บาท ({pct_str})")
    else:
        print("\n➡️ Change:             -")
    
    # แสดงข้อมูลโมเดล
    # **UPDATE: แสดงข้อมูลโมเดลที่จูนแล้ว (Ridge Regressor alpha=100)**
    if 'alpha' in re.sub(r'[^a-zA-Z0-9]', '', metadata['model_type'].lower()):
        model_info_str = f"{metadata['model_type'].upper()} (alpha=100.0 - Tuned)"
    else:
        model_info_str = metadata['model_type'].upper()
        
    print(f"\n🤖 Model Information:")
    print(f"   Type:        {model_info_str}")
    print(f"   Features:    {metadata['feature_count']}")
    print(f"   MAE:         {metadata['metrics']['MAE']:.2f} บาท")
    print(f"   RMSE:        {metadata['metrics']['RMSE']:.2f} บาท")
    print(f"   R²:          {metadata['metrics']['R2']:.4f}")
    print(f"   Trained at:  {metadata['trained_at'][:10]}")
    
    print("\n" + "=" * 60)
    
    # คำแนะนำ
    if isinstance(result['change_pct'], (int, float)) and not np.isnan(result['change_pct']):
        if abs(result['change_pct']) < 0.5:
            print("💡 Prediction: ราคาทองมีแนวโน้มคงที่")
        elif result['change'] > 0:
            print("💡 Prediction: ราคาทองมีแนวโน้มขึ้น")
        else:
            print("💡 Prediction: ราคาทองมีแนวโน้มลง")
    else:
        print("💡 Prediction: —")
    
    print("⚠️  Disclaimer: การทำนายนี้ใช้สำหรับการศึกษาเท่านั้น")
    print("=" * 60 + "\n")

def predict_multiple_days(model, df: pd.DataFrame, feature_cols: list, n_days: int = 7):
    """ทำนายราคาทองหลายวัน (recursive prediction + อัปเดตฟีเจอร์ lag)"""
    predictions = []
    current_df = df.copy()

    print(f"\n🔮 Predicting next {n_days} days...")
    print("=" * 60)

    for i in range(n_days):
        # 1) ทำนายวันถัดไปจากแถวล่าสุด
        result = predict_next_day(model, current_df, feature_cols)
        predictions.append(result)

        print(
            f"Day {i+1}: {result['next_date'].strftime('%Y-%m-%d')} → "
            f"{result['predicted_price']:,.2f} บาท "
            f"({'+' if (not np.isnan(result['change']) and result['change'] > 0) else ''}"
            f"{'' if np.isnan(result['change_pct']) else f'{result['change_pct']:.2f}%'} )"
        )

        # 2) สร้างแถวสำหรับวันถัดไป โดยอัปเดต lag จากค่าทำนายล่าสุด
        last_row = current_df.iloc[-1]
        next_date = result["next_date"]
        new_row = build_next_feature_row(last_row, feature_cols, result["predicted_price"], next_date)

        # 3) แนบเข้า DataFrame เพื่อให้ loop ถัดไปใช้เป็น "แถวล่าสุด"
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

        # 4) ความปลอดภัย: กัน NaN เพิ่มเติม
        current_df = _safe_ffill_zeros(current_df)

    return predictions

def main():
    parser = argparse.ArgumentParser(description="Predict gold price")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, 
                        help="Directory containing model")
    parser.add_argument("--data", type=Path, default=FEATURE_STORE, 
                        help="Path to feature store")
    parser.add_argument("--days", type=int, default=1, 
                        help="Number of days to predict (1-30)")
    parser.add_argument("--save", action="store_true", 
                        help="Save prediction to CSV")
    args = parser.parse_args()
    
    # จำกัดจำนวนวัน
    if args.days < 1 or args.days > 30:
        print("❌ Error: Number of days must be between 1 and 30")
        return
    
    try:
        # โหลดโมเดล
        print("📦 Loading model...")
        # *** NEW: ใช้ load_model_and_metadata ที่ถูกปรับแล้ว ***
        model, metadata = load_model_and_metadata(args.model_dir)
        
        # โหลดข้อมูล
        print("📊 Loading data...")
        df = load_latest_data(args.data)
        print(f"✅ Loaded {len(df)} rows (last: {pd.to_datetime(df.iloc[-1]['date']).strftime('%Y-%m-%d')})")
        
        # ทำนาย
        if args.days == 1:
            result = predict_next_day(model, df, metadata['features'])
            format_output(result, metadata)
            
            # บันทึกผล
            if args.save:
                output_df = pd.DataFrame([{
                    'prediction_date': datetime.now(),
                    'last_date': result['last_date'],
                    'next_date': result['next_date'],
                    'last_price': result['last_price'],
                    'predicted_price': result['predicted_price'],
                    'change': result['change'],
                    'change_pct': result['change_pct']
                }])
                
                output_path = PROJECT_ROOT / "results" / f"prediction_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_df.to_csv(output_path, index=False)
                print(f"💾 Prediction saved to: {output_path}")
        else:
            predictions = predict_multiple_days(model, df, metadata['features'], args.days)
            
            # บันทึกผล
            if args.save:
                output_df = pd.DataFrame([{
                    'date': p['next_date'],
                    'predicted_price': p['predicted_price'],
                    'change': p['change'],
                    'change_pct': p['change_pct']
                } for p in predictions])
                
                output_path = PROJECT_ROOT / "results" / f"predictions_{args.days}days_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_df.to_csv(output_path, index=False)
                print(f"\n💾 Predictions saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n💡 Tip: Run 'python3 model/train_model.py' first to train a model")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()