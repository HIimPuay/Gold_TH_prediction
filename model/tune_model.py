#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_model.py - Hyperparameter Tuning สำหรับโมเดล Ridge Regressor
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================== PATH / CONFIG ==================== #

def find_project_root() -> Path:
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
FEATURE_STORE = PROJECT_ROOT / "data" / "Feature_store" / "feature_store.csv"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"


# ==================== HELPER FUNCTIONS ==================== #

# (สมมติว่าฟังก์ชัน load_data และ prepare_features ถูกคัดลอกมาจาก train_model.py)

def load_data(path: Path) -> pd.DataFrame:
    """โหลดข้อมูลจาก feature store (คัดลอกจาก train_model.py)"""
    # NOTE: ต้องคัดลอก BASE_VARS/BTC_VARS/load_data/prepare_features มาจาก train_model.py
    # ในการรันจริง
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame):
    """เตรียมฟีเจอร์สำหรับโมเดล (คัดลอกจาก train_model.py)"""
    # NOTE: ต้องคัดลอก BASE_VARS/BTC_VARS/prepare_features มาจาก train_model.py
    # ในการรันจริง
    
    # เพื่อให้โค้ดทำงานได้ในบริบทนี้ เราจะใช้ metadata ในการดึง feature_cols
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
        feature_cols = metadata['features']
    else:
        raise FileNotFoundError("❌ model_metadata.pkl not found. Cannot determine feature columns.")
        
    X = df[feature_cols].copy()
    y = df["gold_next"].copy()
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    
    return X[valid_idx], y[valid_idx], feature_cols


def evaluate_model(model, X_test, y_test):
    """ประเมินผลโมเดล"""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ==================== TUNING CONFIG ==================== #

TUNING_MODEL_NAME = "ridge_tuned"

# 🎯 ตัวแปร Hyperparameter ที่จะทำการจูน (Grid Search)
# Alpha คือค่าที่ควบคุมความแข็งแรงของ Regularization (L2 norm)
# ค่าที่สูงขึ้นจะลดความซับซ้อนของโมเดล
PARAM_GRID = {

    # เพิ่มค่า alpha ที่สูงกว่า 100 เข้าไป
    'alpha': [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0]
}

# ==================== MAIN TUNING LOGIC ==================== #

def main():
    print(f"🚀 Starting Hyperparameter Tuning for {TUNING_MODEL_NAME.upper()}")
    print("=" * 60)

    try:
        # 1. โหลดและเตรียมข้อมูล
        df = load_data(FEATURE_STORE)
        X, y, feature_cols = prepare_features(df)

        # Time-ordered split (ใช้ shuffle=False ตามหลัก Time Series)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        print(f"📊 Data split: Train={len(X_train)} | Test={len(X_test)}")

        # 2. ตั้งค่า Grid Search
        ridge_base = Ridge(random_state=42)
        
        # ใช้ scoring เป็น 'neg_mean_absolute_error' เพราะ MAE คือ metric หลัก
        grid_search = GridSearchCV(
            estimator=ridge_base,
            param_grid=PARAM_GRID,
            scoring='neg_mean_absolute_error',
            cv=5, # 5-fold Cross-Validation ในชุดฝึก
            verbose=1,
            n_jobs=-1
        )

        # 3. เริ่มการค้นหา
        print("\n🔍 Starting Grid Search for optimal alpha...")
        grid_search.fit(X_train, y_train)

        # 4. บันทึกและประเมินผล
        best_model = grid_search.best_estimator_
        
        # ประเมินบน Test Set (Unseen data)
        final_metrics = evaluate_model(best_model, X_test, y_test)

        # 5. สรุปผล
        print("\n" + "=" * 60)
        print("✅ TUNING COMPLETE - BEST RESULTS")
        print("=" * 60)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Test Set MAE:  {final_metrics['MAE']:.2f} บาท")
        print(f"Test Set RMSE: {final_metrics['RMSE']:.2f} บาท")
        print(f"Test Set MAPE: {final_metrics['MAPE']:.2f}%")
        
        # 6. บันทึกโมเดลที่จูนแล้ว
        tuned_model_path = MODEL_DIR / f"{TUNING_MODEL_NAME}.pkl"
        joblib.dump(best_model, tuned_model_path)
        print(f"\n💾 Tuned model saved to: {tuned_model_path}")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()