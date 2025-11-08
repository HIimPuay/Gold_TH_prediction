#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py - สร้างโมเดลทำนายราคาทองคำ
รองรับหลายโมเดล: Linear Regression, Random Forest, XGBoost, LightGBM
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------- Optional libs -------------------- #
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("⚠️  XGBoost not installed. Install: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    print("⚠️  LightGBM not installed. Install: pip install lightgbm")

# ==================== PATH / CONFIG ==================== #

def find_project_root() -> Path:
    """หา root directory ของโปรเจกต์ (ให้เทรนได้ไม่ว่าอยู่โฟลเดอร์ไหน)"""
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

# ตัวแปรที่ใช้ (รองรับทั้งมี Bitcoin และไม่มี)
BASE_VARS = ["gold", "fx", "cpi", "oil", "set"]
BTC_VARS = BASE_VARS + ["btc"]

# ==================== FUNCTIONS ==================== #

def load_data(path: Path) -> pd.DataFrame:
    """โหลดข้อมูลจาก feature store"""
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Feature store not found at: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # ตรวจ gold_next
    if "gold_next" not in df.columns:
        raise ValueError("❌ Missing target column 'gold_next' ในไฟล์ feature_store.csv")
    return df

def prepare_features(df: pd.DataFrame):
    """เตรียมฟีเจอร์สำหรับโมเดล"""
    has_btc = "btc" in df.columns
    vars_list = BTC_VARS if has_btc else BASE_VARS
    
    feature_cols = []
    for var in vars_list:
        feature_cols.extend([
            f"{var}_lag1",
            f"{var}_lag3",
            f"{var}_roll7_mean",
            f"{var}_pct_change"
        ])
    feature_cols.extend(vars_list)  # ตัวแปรต้นทาง

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    X = df[feature_cols].copy()
    y = df["gold_next"].copy()

    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    dates = df.loc[valid_idx, "date"]

    print(f"✅ Features prepared: {len(feature_cols)} features, {len(X)} samples")
    print(f"📊 Has Bitcoin: {has_btc}")
    return X, y, dates, feature_cols

def get_models():
    """สร้าง dictionary ของโมเดลทั้งหมด"""
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.1),
        "rf": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            random_state=42
        )
    }
    if HAS_XGB:
        models["xgb"] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
    if HAS_LGB:
        models["lgb"] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    return models

def evaluate_model(model, X_test, y_test):
    """ประเมินผลโมเดล"""
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

def train_and_evaluate(models, X, y, test_size=0.2, random_state=42):
    """เทรนและประเมินโมเดลทั้งหมด (time-ordered split)"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    print(f"\n📊 Data split:\n   Train: {len(X_train)}\n   Test:  {len(X_test)}\n   Test ratio: {test_size*100:.0f}%")

    results, trained_models = {}, {}
    print("\n🔧 Training models...\n" + "=" * 60)

    for name, model in models.items():
        print(f"\n⚙️  Training {name.upper()}...", end=" ")
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=5,
                scoring='neg_mean_absolute_error', n_jobs=-1
            )
            metrics["CV_MAE"] = -cv_scores.mean()
            metrics["CV_STD"] = cv_scores.std()
            results[name] = metrics
            trained_models[name] = model
            print("✅")
            print(f"   MAE:  {metrics['MAE']:.2f} บาท | RMSE: {metrics['RMSE']:.2f} บาท | R²: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.2f}%")
        except Exception as e:
            print(f"❌ Error: {e}")
    return results, trained_models, (X_train, X_test, y_train, y_test)

def save_results(results, feature_cols, output_dir: Path):
    """บันทึกผลลัพธ์การเทียบโมเดล"""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results = pd.DataFrame(results).T.sort_values("MAE")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"model_comparison_{timestamp}.csv"
    df_results.to_csv(results_path)
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(df_results.to_string())
    print(f"\n💾 Results saved to: {results_path}")
    return df_results

def save_best_model(trained_models, results, output_dir: Path, feature_cols):
    """บันทึกโมเดลที่ดีที่สุด (เลือก MAE ต่ำสุด)"""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_name = min(results.items(), key=lambda x: x[1]["MAE"])[0]
    best_model = trained_models[best_name]

    model_path = output_dir / "best_model.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "model_type": best_name,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "metrics": results[best_name],
        "trained_at": datetime.now().isoformat()
    }
    metadata_path = output_dir / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)

    print(f"\n✅ Best model ({best_name.upper()}) saved to: {model_path}")
    print(f"   MAE: {results[best_name]['MAE']:.2f} บาท | RMSE: {results[best_name]['RMSE']:.2f} บาท")
    return best_model, best_name

def plot_predictions(model, X_test, y_test, dates_test, model_name, output_dir: Path):
    """พล็อตผลการทำนาย"""
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        print("⚠️  Matplotlib not available, skipping plots")
        return

    y_pred = model.predict(X_test)

    import matplotlib.dates as mdates
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(dates_test, y_test, label='Actual', linewidth=2)
    ax1.plot(dates_test, y_pred, label='Predicted', linewidth=2, alpha=0.85)
    ax1.fill_between(dates_test, y_test, y_pred, alpha=0.25)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Gold Price (THB)', fontsize=12)
    ax1.set_title(f'Gold Price Prediction - {model_name.upper()}', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator()))

    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.scatter(dates_test, residuals, alpha=0.5)
    ax2.axhline(y=0, linestyle='--', linewidth=2)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Residuals (THB)', fontsize=12)
    ax2.set_title('Prediction Residuals', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"predictions_{model_name}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Prediction plot saved to: {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train gold price prediction model")
    parser.add_argument("--data", type=Path, default=FEATURE_STORE, help="Path to feature store")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Output directory for model")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Output directory for results")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (0-1)")
    parser.add_argument("--plot", action="store_true", help="Generate prediction plots")
    args = parser.parse_args()

    print("🚀 Starting Gold Price Prediction Model Training")
    print("=" * 60)

    # โหลดข้อมูล
    print(f"\n📁 Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # เตรียมฟีเจอร์
    X, y, dates, feature_cols = prepare_features(df)

    # สร้างโมเดล
    models = get_models()
    print(f"\n🎯 Available models: {', '.join(models.keys()).upper()}")

    # เทรนและประเมิน
    results, trained_models, splits = train_and_evaluate(
        models, X, y, test_size=args.test_size
    )

    # บันทึกผลลัพธ์
    df_results = save_results(results, feature_cols, args.results_dir)

    # บันทึกโมเดลที่ดีที่สุด
    best_model, best_name = save_best_model(
        trained_models, results, args.model_dir, feature_cols
    )

    # สร้างกราฟ (ถ้าต้องการ)
    if args.plot:
        X_train, X_test, y_train, y_test = splits
        test_dates = dates.iloc[-len(X_test):].reset_index(drop=True)
        plot_predictions(best_model, X_test, y_test, test_dates, best_name, args.results_dir)

    print("\n✅ Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
