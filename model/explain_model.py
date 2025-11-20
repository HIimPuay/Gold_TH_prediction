#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_model.py - วิเคราะห์ Model Explainability (SHAP)
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# -------------------- Optional libs for Explainability -------------------- #
try:
    import shap
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    # ต้องติดตั้งด้วย pip install shap matplotlib
    print("⚠️ SHAP or Matplotlib not installed. Install: pip install shap matplotlib")

# Keras/TF for loading LSTM models
try:
    from tensorflow import keras
    HAS_TF = True
except Exception:
    HAS_TF = False

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

# ==================== CORE FUNCTIONS ==================== #

def load_data_and_model(data_path: Path):
    """
    โหลดโมเดลที่ดีที่สุด, metadata, และข้อมูล X_test ที่ใช้ในการทำนาย (สำหรับ SHAP)
    """
    
    # 1. Load Metadata
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Metadata not found at: {metadata_path}. Run train_model.py first!")
    metadata = joblib.load(metadata_path)
    model_type = metadata.get('model_type', 'unknown')
    feature_cols = metadata['features']
    
    # 2. Load Model
    if model_type == "lstm":
        if not HAS_TF:
            raise ImportError("TensorFlow/Keras is required to load LSTM model.")
        model_path = MODEL_DIR / "best_model_lstm.keras"
        if not model_path.exists():
             raise FileNotFoundError(f"❌ LSTM Model not found at: {model_path}. Run train_model.py first!")
        best_model = keras.models.load_model(model_path)
    else:
        model_path = MODEL_DIR / "best_model.pkl"
        if not model_path.exists():
             raise FileNotFoundError(f"❌ SKLearn/Base Model not found at: {model_path}. Run train_model.py first!")
        best_model = joblib.load(model_path)

    print(f"✅ Loaded Best Model: {model_type.upper()}")
    
    # 3. Load Data (Re-construct X_test based on train_model.py's logic)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ Feature store not found at: {data_path}")
    df_raw = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    
    # Replicate prepare_features logic from train_model.py
    X = df_raw[feature_cols].copy()
    y = df_raw["gold_next"].copy()
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X_skl = X[valid_idx]

    # Replicate train_test_split (test_size=0.2, shuffle=False)
    test_size = 0.2
    training_data_len = int(np.ceil(len(X_skl) * (1 - test_size)))
    X_test = X_skl.iloc[training_data_len:]
    
    print(f"✅ Loaded X_test data: {len(X_test)} samples")

    return best_model, X_test, feature_cols, model_type


def explain_model_shap(best_model, X_test: pd.DataFrame, feature_cols: list, model_name: str, output_dir: Path):
    """
    คำนวณ SHAP values และพล็อต summary/dependence plots สำหรับโมเดลที่ดีที่สุด
    """
    if not HAS_SHAP:
        return
    
    # ข้าม LSTM เพราะต้องใช้ DeepExplainer และข้อมูล 3D
    if model_name == "lstm":
        print(f"\n⚠️ Skipping SHAP for {model_name.upper()} as it requires specialized DeepExplainer/3D data handling.")
        return

    print("\n🔬 Starting SHAP Explainability...")

    # ตรวจสอบประเภทโมเดลเพื่อเลือก Explainer ที่เหมาะสม
    if "xgb" in model_name or "lgb" in model_name or "rf" in model_name or "gbm" in model_name:
        explainer = shap.TreeExplainer(best_model)
    elif "linear" in model_name or "ridge" in model_name or "lasso" in model_name:
        explainer = shap.LinearExplainer(best_model, X_test)
    else:
        print("   Using KernelExplainer (Slow)...")
        # ใช้ subset ของข้อมูลทดสอบเพื่อความเร็ว (จำกัดที่ 50 ตัวอย่าง)
        X_test_subset = X_test.sample(n=min(50, len(X_test)), random_state=42)
        explainer = shap.KernelExplainer(best_model.predict, X_test_subset)
        X_test = X_test_subset
        
    X_test_for_shap = X_test 

    try:
        print(f"   Calculating SHAP values for {len(X_test_for_shap)} samples...")
        shap_values = explainer.shap_values(X_test_for_shap)
    except Exception as e:
        print(f"❌ Error calculating SHAP values: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot 1: Summary Plot (Bar - Feature Importance)
    print("   Generating SHAP Bar Plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_for_shap, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance - {model_name.upper()}')
    plt.tight_layout()
    plot_path_bar = output_dir / f"shap_summary_bar_{model_name}_{timestamp}.png"
    plt.savefig(plot_path_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 SHAP Bar Plot saved to: {plot_path_bar}")


    # Plot 2: Summary Plot (Dot - Global Feature Importance + Impact)
    print("   Generating SHAP Dot Plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    # สำหรับ Regression, shap_values อาจจะเป็น array หรือ list ที่มี array เดียว (สำหรับ Tree-based)
    if isinstance(shap_values, list):
         shap.summary_plot(shap_values[0], X_test_for_shap, feature_names=feature_cols, show=False)
    else:
         shap.summary_plot(shap_values, X_test_for_shap, feature_names=feature_cols, show=False)
    
    plt.title(f'SHAP Summary Plot - {model_name.upper()}')
    plt.tight_layout()
    plot_path_dot = output_dir / f"shap_summary_dot_{model_name}_{timestamp}.png"
    plt.savefig(plot_path_dot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 SHAP Dot Plot saved to: {plot_path_dot}")
    
    print("✅ SHAP Explainability complete.")


def main():
    parser = argparse.ArgumentParser(description="Analyze Model Explainability (SHAP) for the best model")
    parser.add_argument("--data", type=Path, default=FEATURE_STORE, help="Path to feature store (required to reconstruct X_test)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Output directory for SHAP plots")
    args = parser.parse_args()
    
    if not HAS_SHAP:
        print("Cannot run explainability. Please install SHAP and Matplotlib: pip install shap matplotlib")
        return

    print("🚀 Starting Model Explainability (SHAP) Analysis")
    print("=" * 60)
    
    try:
        # โหลดโมเดล ข้อมูลทดสอบ และ metadata
        best_model, X_test, feature_cols, model_name = load_data_and_model(args.data)

        # คำนวณและพล็อต SHAP
        explain_model_shap(best_model, X_test, feature_cols, model_name, args.results_dir)

    except FileNotFoundError as e:
        print(f"\n{e}")
    except ImportError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        
    print("=" * 60)

if __name__ == "__main__":
    main()