#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gold_skip_sundays.py - ทำนายราคาทองโดยข้ามวันอาทิตย์
เพิ่มความสามารถในการข้ามวันที่ตลาดปิด
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Import config
try:
    from gold_config import (
        GOLD_PRICE_TYPE, 
        MARKET_CLOSED_DAYS, 
        PREDICTION_DAYS,
        SKIP_CLOSED_DAYS_IN_PREDICTION
    )
except ImportError:
    GOLD_PRICE_TYPE = "gold_bar_sell"
    MARKET_CLOSED_DAYS = [6]  # Sunday
    PREDICTION_DAYS = 7
    SKIP_CLOSED_DAYS_IN_PREDICTION = True

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
RESULTS_DIR = PROJECT_ROOT / "results"

def is_market_open(date):
    """ตรวจสอบว่าตลาดเปิดในวันนี้หรือไม่"""
    weekday = date.weekday()
    return weekday not in MARKET_CLOSED_DAYS

def get_next_business_date(date):
    """หาวันทำการถัดไป"""
    next_date = date + timedelta(days=1)
    while not is_market_open(next_date):
        next_date += timedelta(days=1)
    return next_date

def load_model_and_metadata(model_dir: Path):
    """โหลดโมเดลและ metadata"""
    model_path = model_dir / "best_model.pkl"
    metadata_path = model_dir / "model_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found at: {model_path}")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Metadata not found at: {metadata_path}")
    
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
    return df.ffill().fillna(0)

def predict_next_day(model, df: pd.DataFrame, feature_cols: list):
    """ทำนายราคาทองวันถัดไป"""
    latest = df.iloc[-1:].copy()
    
    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    X = latest[feature_cols].copy()
    
    if X.isna().any().any():
        print("⚠️  Warning: Found NaN in features, filling by ffill/0")
        X = _safe_ffill_zeros(X)
    
    prediction = float(model.predict(X)[0])
    
    last_date = pd.to_datetime(df.iloc[-1]["date"])
    last_gold = float(df.iloc[-1]["gold"]) if "gold" in df.columns else np.nan
    
    if np.isnan(last_gold):
        change = np.nan
        change_pct = np.nan
    else:
        change = prediction - last_gold
        change_pct = (change / last_gold) * 100 if last_gold != 0 else np.nan
    
    # หาวันทำการถัดไป
    next_date = get_next_business_date(last_date)
    
    return {
        "last_date": last_date,
        "last_price": last_gold,
        "predicted_price": prediction,
        "change": change,
        "change_pct": change_pct,
        "next_date": next_date,
        "is_business_day": is_market_open(next_date)
    }

def build_next_feature_row(last_row: pd.Series, feature_cols: list, 
                          predicted_price: float, next_date: pd.Timestamp) -> pd.Series:
    """สร้างแถวฟีเจอร์สำหรับวันถัดไป"""
    new_row = last_row.copy()
    new_row["date"] = pd.to_datetime(next_date)
    
    if "gold" in new_row.index:
        new_row["gold"] = float(predicted_price)
    
    # อัพเดท lag features
    for col in last_row.index:
        if "_lag" in col:
            parts = col.split("_lag")
            if len(parts) == 2:
                base = parts[0]
                try:
                    n = int(parts[1])
                    if base == "gold":
                        if n == 1:
                            new_row[col] = float(last_row.get("gold", np.nan))
                        else:
                            prev_col = f"{base}_lag{n-1}"
                            if prev_col in last_row.index:
                                new_row[col] = last_row.get(prev_col, np.nan)
                    else:
                        if n == 1:
                            new_row[col] = last_row.get(base, np.nan)
                        else:
                            prev_col = f"{base}_lag{n-1}"
                            new_row[col] = last_row.get(prev_col, last_row.get(base, np.nan))
                except ValueError:
                    pass
    
    new_row = new_row.ffill().fillna(0)
    return new_row

def predict_multiple_days(model, df: pd.DataFrame, feature_cols: list, n_days: int = 7):
    """ทำนายราคาทองหลายวัน (ข้ามวันที่ตลาดปิด)"""
    predictions = []
    current_df = df.copy()
    
    print(f"\n🔮 Predicting next {n_days} business days...")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    closed_days_str = ', '.join([day_names[d] for d in MARKET_CLOSED_DAYS])
    print(f"   (Skipping: {closed_days_str})")
    print("=" * 70)
    
    actual_predictions = 0
    
    while actual_predictions < n_days:
        result = predict_next_day(model, current_df, feature_cols)
        
        # ถ้าเป็นวันทำการ หรือไม่ต้องข้าม ให้นับเข้าไป
        if result["is_business_day"] or not SKIP_CLOSED_DAYS_IN_PREDICTION:
            actual_predictions += 1
            predictions.append(result)
            
            status = "📈" if (not np.isnan(result['change']) and result['change'] > 0) else \
                    "📉" if (not np.isnan(result['change']) and result['change'] < 0) else "➡️"
            
            day_name = result['next_date'].strftime('%A')
            change_str = 'N/A' if np.isnan(result['change_pct']) else f"{result['change_pct']:+.2f}%"
            print(
                f"Day {actual_predictions}: {result['next_date'].strftime('%Y-%m-%d')} ({day_name}) {status} "
                f"{result['predicted_price']:,.2f} บาท "
                f"({change_str})"
            )
        
        # สร้างแถวใหม่สำหรับการทำนายครั้งถัดไป
        last_row = current_df.iloc[-1]
        new_row = build_next_feature_row(
            last_row, feature_cols, 
            result["predicted_price"], 
            result["next_date"]
        )
        
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
        current_df = _safe_ffill_zeros(current_df)
    
    return predictions

def format_output(result: dict, metadata: dict):
    """จัดรูปแบบการแสดงผล"""
    print("\n" + "=" * 70)
    print("🔮 GOLD PRICE PREDICTION")
    print("=" * 70)
    print(f"\n📊 Using: {GOLD_PRICE_TYPE}")
    print(f"📅 Last available date: {result['last_date'].strftime('%Y-%m-%d')}")
    
    if not np.isnan(result['last_price']):
        print(f"💰 Last gold price:     {result['last_price']:,.2f} บาท")
    else:
        print(f"💰 Last gold price:     -")
    
    print(f"\n📅 Prediction for:      {result['next_date'].strftime('%Y-%m-%d')} ({result['next_date'].strftime('%A')})")
    print(f"💎 Predicted price:     {result['predicted_price']:,.2f} บาท")
    
    if not np.isnan(result['change']):
        change_symbol = "📈" if result['change'] > 0 else "📉" if result['change'] < 0 else "➡️"
        sign = "+" if result['change'] > 0 else ""
        pct_str = f"{sign}{result['change_pct']:.2f}%" if result['change_pct'] == result['change_pct'] else "-"
        print(f"\n{change_symbol} Change:             {sign}{result['change']:,.2f} บาท ({pct_str})")
    
    print(f"\n🤖 Model Information:")
    print(f"   Type:        {metadata['model_type'].upper()}")
    print(f"   Features:    {metadata['feature_count']}")
    print(f"   MAE:         {metadata['metrics']['MAE']:.2f} บาท")
    print(f"   RMSE:        {metadata['metrics']['RMSE']:.2f} บาท")
    print(f"   R²:          {metadata['metrics']['R2']:.4f}")
    
    print("\n" + "=" * 70)
    print("⚠️  คำเตือน: การทำนายนี้ใช้สำหรับการศึกษาเท่านั้น")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Predict gold price (skip market closed days)")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data", type=Path, default=FEATURE_STORE)
    parser.add_argument("--days", type=int, default=PREDICTION_DAYS,
                       help="Number of business days to predict")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    
    if args.days < 1 or args.days > 30:
        print("❌ Error: Number of days must be between 1 and 30")
        return
    
    try:
        print("📦 Loading model...")
        model, metadata = load_model_and_metadata(args.model_dir)
        print(f"✅ Loaded {metadata['model_type'].upper()} model")
        
        print("📊 Loading data...")
        df = load_latest_data(args.data)
        print(f"✅ Loaded {len(df)} rows (last: {pd.to_datetime(df.iloc[-1]['date']).strftime('%Y-%m-%d')})")
        
        if args.days == 1:
            result = predict_next_day(model, df, metadata['features'])
            format_output(result, metadata)
            
            if args.save:
                output_df = pd.DataFrame([{
                    'prediction_date': datetime.now(),
                    'target_date': result['next_date'],
                    'predicted_price': result['predicted_price'],
                    'price_type': GOLD_PRICE_TYPE,
                    'change': result['change'],
                    'change_pct': result['change_pct'],
                    'is_business_day': result['is_business_day']
                }])
                
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                output_path = RESULTS_DIR / f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_df.to_csv(output_path, index=False)
                print(f"💾 Prediction saved to: {output_path}")
        else:
            predictions = predict_multiple_days(model, df, metadata['features'], args.days)
            
            if args.save:
                output_df = pd.DataFrame([{
                    'date': p['next_date'],
                    'day_name': p['next_date'].strftime('%A'),
                    'predicted_price': p['predicted_price'],
                    'price_type': GOLD_PRICE_TYPE,
                    'change': p['change'],
                    'change_pct': p['change_pct'],
                    'is_business_day': p['is_business_day']
                } for p in predictions])
                
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                output_path = RESULTS_DIR / f"predictions_{args.days}days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_df.to_csv(output_path, index=False)
                print(f"\n💾 Predictions saved to: {output_path}")
                print(f"   Total business days predicted: {len(predictions)}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\n💡 Tip: Run 'python3 train_model.py' first")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()