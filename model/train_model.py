#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py - สร้างโมเดลทำนายราคาทองคำ
รองรับหลายโมเดล: Linear Regression, Random Forest, XGBoost, LightGBM, LSTM
✅ แก้ไข: รองรับชื่อคอลัมน์ _roll7_mean และ _pct_change
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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# --- TensorFlow/Keras imports for LSTM ---
try:
    from tensorflow import keras
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except Exception:
    HAS_TF = False
    print("⚠️  TensorFlow/Keras not installed. Install: pip install tensorflow scikit-learn")

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

# ตัวแปรที่ใช้
BASE_VARS = ["gold", "fx", "cpi", "oil", "set"]
BTC_VARS = BASE_VARS + ["btc"]

# ==================== FUNCTIONS ==================== #

def load_data(path: Path) -> pd.DataFrame:
    """โหลดข้อมูลจาก feature store"""
    if not Path(path).exists():
        raise FileNotFoundError(f"❌ Feature store not found at: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if "gold_next" not in df.columns:
        raise ValueError("❌ Missing target column 'gold_next'")
    return df

def prepare_features(df: pd.DataFrame):
    """เตรียมฟีเจอร์สำหรับโมเดล - รองรับทั้งชื่อเก่าและชื่อใหม่"""
    has_btc = "btc" in df.columns
    vars_list = BTC_VARS if has_btc else BASE_VARS
    
    # ✅ ตรวจสอบว่าใช้ชื่อแบบไหน
    # ชื่อใหม่: _roll7_mean, _pct_change
    # ชื่อเก่า: _roll7, _pct
    use_new_names = "gold_roll7_mean" in df.columns
    
    feature_cols = []
    for var in vars_list:
        feature_cols.extend([
            f"{var}_lag1",
            f"{var}_lag3",
            f"{var}_roll7_mean" if use_new_names else f"{var}_roll7",
            f"{var}_pct_change" if use_new_names else f"{var}_pct"
        ])
    feature_cols.extend(vars_list)
    
    # ตรวจสอบคอลัมน์ที่หาไม่เจอ
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        
        # ถ้าหาไม่เจอ ลองใช้ชื่อแบบอีกแบบ
        if use_new_names:
            print("   Trying old column names (_roll7, _pct)...")
            feature_cols_alt = []
            for var in vars_list:
                feature_cols_alt.extend([
                    f"{var}_lag1",
                    f"{var}_lag3",
                    f"{var}_roll7",
                    f"{var}_pct"
                ])
            feature_cols_alt.extend(vars_list)
            
            missing_alt = [c for c in feature_cols_alt if c not in df.columns]
            if not missing_alt:
                print("   ✅ Found old column names!")
                feature_cols = feature_cols_alt
                use_new_names = False
            else:
                raise ValueError(f"Missing features: {missing}")
        else:
            raise ValueError(f"Missing features: {missing}")
    
    X = df[feature_cols].copy()
    y = df["gold_next"].copy()
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    dates = df.loc[valid_idx, "date"]
    
    col_suffix = "_mean/_change" if use_new_names else "_roll7/_pct"
    print(f"✅ Features prepared: {len(feature_cols)} features, {len(X)} samples")
    print(f"📊 Has Bitcoin: {has_btc}")
    print(f"📋 Column naming: {col_suffix}")
    
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
    
    if HAS_TF:
        models["lstm"] = LinearRegression()  # Placeholder
    
    return models

def evaluate_model(model, X_test, y_test):
    """ประเมินผลโมเดล"""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2, "MAPE": mape}

def train_and_evaluate(models, X, y, test_size=0.2, random_state=42):
    """เทรนและประเมินโมเดลทั้งหมด"""
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

# -------------------- LSTM FUNCTIONS -------------------- #

def prepare_lstm_data(df: pd.DataFrame, feature_cols: list, TIME_STEP: int, test_size: float):
    """เตรียมข้อมูลสำหรับ LSTM"""
    data_for_lstm = df[feature_cols].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_lstm)
    
    dataset_len = len(df)
    training_data_len = int(np.ceil(dataset_len * (1 - test_size)))
    
    training_data = scaled_data[:training_data_len]
    test_data_split = scaled_data[training_data_len - TIME_STEP:]
    
    X_train_lstm, Y_train_lstm = [], []
    for i in range(TIME_STEP, len(training_data)):
        X_train_lstm.append(training_data[i - TIME_STEP:i, :])
        Y_train_lstm.append(training_data[i, 0])
    
    X_train_lstm, Y_train_lstm = np.array(X_train_lstm), np.array(Y_train_lstm)
    
    X_test_lstm = []
    for i in range(TIME_STEP, len(test_data_split)):
        X_test_lstm.append(test_data_split[i - TIME_STEP:i, :])
    
    X_test_lstm = np.array(X_test_lstm)
    Y_test_actual = df["gold_next"].values[training_data_len:]
    
    return X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len

def build_lstm_model(input_shape):
    """สร้างโมเดล LSTM"""
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(64, return_sequences=False))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(1))
    return model

def train_and_evaluate_lstm(df: pd.DataFrame, feature_cols: list, test_size: float):
    """เทรนและประเมิน LSTM"""
    TIME_STEP = 60
    MODEL_NAME = "lstm"
    
    print("\n" + "=" * 60)
    print(f"🔧 Starting {MODEL_NAME.upper()} Model (TIME_STEP={TIME_STEP}) Training...")
    
    try:
        X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len = \
            prepare_lstm_data(df, feature_cols, TIME_STEP, test_size)
        
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        lstm_model = build_lstm_model(input_shape)
        
        lstm_model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
        
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
        
        print("   Training LSTM for up to 300 epochs...", end=" ")
        history = lstm_model.fit(
            X_train_lstm, Y_train_lstm,
            epochs=300, batch_size=32, verbose=0,
            validation_split=0.1,
            callbacks=[es]
        )
        print("✅")
        
        predictions_scaled = lstm_model.predict(X_test_lstm, verbose=0)
        
        n_features = scaler.n_features_in_
        predictions_dummy = np.zeros((predictions_scaled.shape[0], n_features))
        predictions_dummy[:, 0] = predictions_scaled.flatten()
        predictions_unscaled = scaler.inverse_transform(predictions_dummy)[:, 0].flatten()
        
        start_index = len(Y_test_actual) - len(predictions_unscaled)
        Y_test_actual_trimmed = Y_test_actual[start_index:]
        
        mae = mean_absolute_error(Y_test_actual_trimmed, predictions_unscaled)
        mse = mean_squared_error(Y_test_actual_trimmed, predictions_unscaled)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((Y_test_actual_trimmed - predictions_unscaled) / Y_test_actual_trimmed)) * 100
        
        lstm_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": None, "MAPE": mape, "CV_MAE": None, "CV_STD": None}
        
        print(f"   MAE:  {lstm_metrics['MAE']:.2f} บาท | RMSE: {lstm_metrics['RMSE']:.2f} บาท | MAPE: {lstm_metrics['MAPE']:.2f}%")
        
        return {MODEL_NAME: lstm_metrics}, {MODEL_NAME: lstm_model}, MODEL_NAME
    
    except Exception as e:
        print(f"❌ Error during {MODEL_NAME.upper()} training: {e}")
        return {}, {}, None

def get_predictions(model, name, X_test, y_test, df_full, feature_cols, test_size):
    """ทำนายและคืนค่า y_pred"""
    if name != "lstm":
        y_pred = model.predict(X_test)
        y_actual = y_test.values
    else:
        TIME_STEP = 60
        df_lstm_for_plot = df_full.loc[X_test.index].copy()
        
        X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len = \
            prepare_lstm_data(df_lstm_for_plot, feature_cols, TIME_STEP, test_size)
        
        predictions_scaled = model.predict(X_test_lstm, verbose=0)
        
        n_features = scaler.n_features_in_
        predictions_dummy = np.zeros((predictions_scaled.shape[0], n_features))
        predictions_dummy[:, 0] = predictions_scaled.flatten()
        predictions_unscaled = scaler.inverse_transform(predictions_dummy)[:, 0].flatten()
        
        start_index_trim = len(Y_test_actual) - len(predictions_unscaled)
        y_actual = Y_test_actual[start_index_trim:]
        y_pred = predictions_unscaled
    
    return y_actual, y_pred

# -------------------- SAVE & PLOT -------------------- #

def save_results(results, feature_cols, output_dir: Path):
    """บันทึกผลลัพธ์"""
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
    """บันทึกโมเดลที่ดีที่สุด"""
    output_dir.mkdir(parents=True, exist_ok=True)
    best_name = min(results.items(), key=lambda x: x[1]["MAE"])[0]
    best_model = trained_models[best_name]
    
    if best_name == "lstm":
        model_path = output_dir / "best_model_lstm.keras"
        best_model.save(model_path)
        print(f"\n✅ Best model ({best_name.upper()}) saved (Keras format) to: {model_path}")
    else:
        model_path = output_dir / "best_model.pkl"
        joblib.dump(best_model, model_path)
        print(f"\n✅ Best model ({best_name.upper()}) saved (Joblib format) to: {model_path}")
    
    metadata = {
        "model_type": best_name,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "metrics": results[best_name],
        "trained_at": datetime.now().isoformat()
    }
    metadata_path = output_dir / "model_metadata.pkl"
    joblib.dump(metadata, metadata_path)
    
    print(f"   MAE: {results[best_name]['MAE']:.2f} บาท | RMSE: {results[best_name]['RMSE']:.2f} บาท")
    return best_model, best_name

def plot_predictions(trained_models, df_full, feature_cols, splits, dates_full, test_size, output_dir):
    """เปรียบเทียบผลทำนาย"""
    X_train, X_test, y_train, y_test = splits
    dates_test = dates_full.loc[X_test.index].reset_index(drop=True)
    df_plot = pd.DataFrame({'date': dates_test, 'Actual': y_test.values})
    
    print("\n📈 Generating prediction comparison plot...")
    
    for name, model in trained_models.items():
        y_actual_trimmed, y_pred = get_predictions(
            model, name, X_test, y_test, df_full, feature_cols, test_size
        )
        
        if name != "lstm":
            df_plot[f'{name.upper()}_Pred'] = y_pred
        else:
            start_index_trim = len(df_plot) - len(y_pred)
            df_lstm_plot = pd.DataFrame({
                'date': dates_test[start_index_trim:].reset_index(drop=True),
                f'{name.upper()}_Pred': y_pred
            })
            df_plot = pd.merge(df_plot, df_lstm_plot, on='date', how='left')
        
        print(f"   {name.upper()} predictions added (Length: {len(y_pred)})")
    
    plt.style.use('ggplot')
    fig_width = max(15, len(df_plot) / 15)
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    ax.plot(df_plot['date'], df_plot['Actual'], label='Actual Gold Price',
            color='black', linewidth=3, alpha=0.9)
    
    model_preds = [col for col in df_plot.columns if '_Pred' in col]
    color_mapping = {
        'RIDGE_Pred': 'blue',
        'LINEAR_Pred': 'green',
        'LASSO_Pred': 'purple',
        'RF_Pred': 'orange',
        'GBM_Pred': 'brown',
        'LSTM_Pred': 'red',
    }
    
    colors_map = plt.cm.get_cmap('tab10')
    
    for i, col in enumerate(model_preds):
        plot_color = color_mapping.get(col, colors_map(i))
        ax.plot(df_plot['date'], df_plot[col],
                label=col.replace('_Pred', ''),
                color=plot_color,
                linestyle='--',
                alpha=0.7)
    
    ax.set_title(f'Gold Price Prediction Comparison on Test Set (Test Size: {test_size*100:.0f}%)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Gold Price (THB)', fontsize=12)
    ax.legend(loc='best', fontsize=10, ncol=min(3, len(model_preds)+1))
    ax.grid(True, linestyle=':', alpha=0.6)
    
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=45)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"predictions_comparison_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"💾 Prediction comparison plot saved to: {plot_path}")
    return df_plot

# ==================== MAIN ==================== #

def main(args):
    print("🚀 Starting Gold Price Prediction Model Training")
    print("=" * 60)
    
    print(f"\n📁 Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"   Loaded {len(df)} rows")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    X, y, dates, feature_cols = prepare_features(df)
    
    models = get_models()
    lstm_placeholder_model = models.pop("lstm", None)
    
    print(f"\n🎯 Available models: {', '.join(models.keys()).upper()}")
    
    results, trained_models, splits = train_and_evaluate(
        models, X, y, test_size=args.test_size
    )
    
    if HAS_TF and lstm_placeholder_model:
        df_lstm = df.loc[X.index].copy()
        lstm_results, lstm_trained_models, _ = train_and_evaluate_lstm(
            df_lstm, feature_cols, args.test_size
        )
        results.update(lstm_results)
        trained_models.update(lstm_trained_models)
    
    df_results = save_results(results, feature_cols, args.results_dir)
    best_model, best_name = save_best_model(
        trained_models, results, args.model_dir, feature_cols
    )
    
    if args.plot:
        df_predictions = plot_predictions(
            trained_models, df, feature_cols, splits, dates,
            args.test_size, args.results_dir
        )
    
    print("\n✅ Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gold price prediction model")
    parser.add_argument("--data", type=Path, default=FEATURE_STORE)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    
    main(args)