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
from tensorflow.keras.callbacks import EarlyStopping

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

    # Add LSTM placeholder
    if HAS_TF:
        # ใช้ LinearRegression เป็น placeholder แต่จะเทรนจริงในฟังก์ชันแยก
        models["lstm"] = LinearRegression()

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

# เพิ่มฟังก์ชันนี้ต่อจากส่วน evaluate_model(model, X_test, y_test)

def get_predictions(model, name, X_test, y_test, df_full, feature_cols, test_size):
    """ทำนายและคืนค่า y_pred สำหรับโมเดลทั้งหมด"""
    if name != "lstm":
        # สำหรับโมเดล Scikit-learn (Linear, RF, XGB, etc.)
        y_pred = model.predict(X_test)
        # ใช้ y_test จาก splits
        y_actual = y_test.values 
    else:
        # สำหรับโมเดล LSTM (ต้องใช้การเตรียมข้อมูลเฉพาะ)
        TIME_STEP = 60
        df_lstm_for_plot = df_full.loc[X_test.index].copy() 

        X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len = \
            prepare_lstm_data(df_lstm_for_plot, feature_cols, TIME_STEP, test_size)
            
        predictions_scaled = model.predict(X_test_lstm, verbose=0)

        # Inverse Transform
        n_features = scaler.n_features_in_
        predictions_dummy = np.zeros((predictions_scaled.shape[0], n_features))
        predictions_dummy[:, 0] = predictions_scaled.flatten() 
        predictions_unscaled = scaler.inverse_transform(predictions_dummy)[:, 0].flatten()

        # Trim Actuals and Predictions to match the effective prediction window
        start_index_trim = len(Y_test_actual) - len(predictions_unscaled)
        y_actual = Y_test_actual[start_index_trim:]
        y_pred = predictions_unscaled

    return y_actual, y_pred

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

# -------------------- LSTM FUNCTIONS -------------------- #

def prepare_lstm_data(df: pd.DataFrame, feature_cols: list, TIME_STEP: int, test_size: float):
    """เตรียมข้อมูล (Scaling + Sliding Window) สำหรับ LSTM"""
    
    # 1. Cleaning and Scaling
    data_for_lstm = df[feature_cols].values 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_for_lstm)

    # 2. Split (Time-based split, using the same ratio as sklearn split)
    dataset_len = len(df)
    training_data_len = int(np.ceil(dataset_len * (1 - test_size)))

    training_data = scaled_data[:training_data_len] 
    # ต้องรวม TIME_STEP วันสุดท้ายของ Train set เพื่อใช้สร้าง Window แรกของ Test set
    test_data_split = scaled_data[training_data_len - TIME_STEP:] 

    # 3. Create Sliding Window for Train Set
    X_train_lstm, Y_train_lstm = [], []
    for i in range(TIME_STEP, len(training_data)):
        X_train_lstm.append(training_data[i - TIME_STEP:i, :]) 
        Y_train_lstm.append(training_data[i, 0]) # Target is 'gold' (index 0)

    X_train_lstm, Y_train_lstm = np.array(X_train_lstm), np.array(Y_train_lstm)

    # 4. Create Sliding Window for Test Set
    X_test_lstm = []
    for i in range(TIME_STEP, len(test_data_split)):
        X_test_lstm.append(test_data_split[i - TIME_STEP:i, :])
    
    X_test_lstm = np.array(X_test_lstm)

    # 5. Extract unscaled actual Y_test for metrics calculation
    # Y_test_actual: ราคาทองคำจริง (Unscaled) ของ Test Set
    Y_test_actual = df["gold_next"].values[training_data_len:]

    return X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len


def build_lstm_model(input_shape):
    """สร้างโมเดล LSTM แบบ Multivariate"""
    model = keras.models.Sequential()
    
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(64, return_sequences=False))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(1)) 

    return model


def train_and_evaluate_lstm(df: pd.DataFrame, feature_cols: list, test_size: float) -> tuple[dict, dict, str]:
    """เทรนและประเมินโมเดล LSTM โดยเฉพาะ"""

    TIME_STEP = 60 # ใช้ข้อมูลย้อนหลัง 60 วัน
    MODEL_NAME = "lstm"
    
    print("\n" + "=" * 60)
    print(f"🔧 Starting {MODEL_NAME.upper()} Model (TIME_STEP={TIME_STEP}) Training...")
    
    try:
        X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len = \
            prepare_lstm_data(df, feature_cols, TIME_STEP, test_size)
        
        input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
        lstm_model = build_lstm_model(input_shape)

        # 1. Compile and Train
        lstm_model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
        
        # *** เพิ่ม Early Stopping เพื่อความเป็นกลาง ***
        # ใช้ patience 20 เพื่อให้โมเดลหยุดเมื่อ loss ไม่ลดลงเกิน 20 epochs
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

        print("   Training LSTM for 150 epochs...", end=" ")
        history = lstm_model.fit(
            X_train_lstm, 
            Y_train_lstm, 
            epochs=300, # เพิ่ม Epochs สูงสุด เพื่อให้ Early Stopping ทำงาน
            batch_size=32, 
            verbose=0,
            validation_split=0.1, # ใช้ 10% ของ Training Data เป็น Validation Set
            callbacks=[es]
        )
        print("✅")

        # 2. Predict
        predictions_scaled = lstm_model.predict(X_test_lstm, verbose=0)

        # 3. Inverse Transform (Requires a dummy array of size n_features)
        n_features = scaler.n_features_in_
        predictions_dummy = np.zeros((predictions_scaled.shape[0], n_features))
        predictions_dummy[:, 0] = predictions_scaled.flatten() # Gold is at index 0

        predictions_unscaled = scaler.inverse_transform(predictions_dummy)[:, 0].flatten()

        # 4. Trim Actuals to match predictions size
        start_index = len(Y_test_actual) - len(predictions_unscaled)
        Y_test_actual_trimmed = Y_test_actual[start_index:]
        
        # 5. Calculate Metrics
        mae  = mean_absolute_error(Y_test_actual_trimmed, predictions_unscaled)
        mse  = mean_squared_error(Y_test_actual_trimmed, predictions_unscaled)
        rmse = np.sqrt(mse)
        # R2 และ CV_MAE/STD ไม่สามารถคำนวณได้โดยตรง
        mape = np.mean(np.abs((Y_test_actual_trimmed - predictions_unscaled) / Y_test_actual_trimmed)) * 100
        
        lstm_metrics = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": None, "MAPE": mape, "CV_MAE": None, "CV_STD": None}
        
        print(f"   MAE:  {lstm_metrics['MAE']:.2f} บาท | RMSE: {lstm_metrics['RMSE']:.2f} บาท | MAPE: {lstm_metrics['MAPE']:.2f}%")
        
        # Return in the same format as the main training function
        return {MODEL_NAME: lstm_metrics}, {MODEL_NAME: lstm_model}, MODEL_NAME

    except Exception as e:
        print(f"❌ Error during {MODEL_NAME.upper()} training/evaluation: {e}")
        # หากเกิด Error ให้ return dictionary ว่าง
        return {}, {}, None

# -------------------- END LSTM FUNCTIONS -------------------- #

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

    # model_path = output_dir / "best_model.pkl"
    # joblib.dump(best_model, model_path)

    # บันทึกโมเดล
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

    # LSTM models have incompatible X_test (3D) and y_test (unscaled), so we skip plotting for them here.
    # if model_name == "lstm":
    #      print(f"⚠️  Skipping plotting for {model_name.upper()} due to incompatible data format.")
    #      return

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
    plt.show() 
    plt.close()


# -------------------- PLOTTING LSTM FUNCTIONS -------------------- #

def plot_lstm_predictions(lstm_model, df: pd.DataFrame, feature_cols: list, test_size: float, output_dir: Path):
    """
    พล็อตผลการทำนายสำหรับ LSTM โดยเฉพาะ (เนื่องจากต้องเตรียมข้อมูลแตกต่างจาก sklearn)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        print("⚠️  Matplotlib not available, skipping plots")
        return

    MODEL_NAME = "lstm"
    TIME_STEP = 60 # ต้องตรงกับค่าที่ใช้ในการเทรน
    
    print(f"📈 Generating plot for {MODEL_NAME.upper()}...")

    # 1. Redo data preparation to get Scaler, X_test_lstm, and Y_test_actual
    X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_actual, scaler, training_data_len = \
        prepare_lstm_data(df, feature_cols, TIME_STEP, test_size)
        
    # 2. Predict
    predictions_scaled = lstm_model.predict(X_test_lstm, verbose=0)

    # 3. Inverse Transform 
    n_features = scaler.n_features_in_
    predictions_dummy = np.zeros((predictions_scaled.shape[0], n_features))
    predictions_dummy[:, 0] = predictions_scaled.flatten() 
    predictions_unscaled = scaler.inverse_transform(predictions_dummy)[:, 0].flatten()

    # 4. Extract dates
    # Test set starts from training_data_len (index)
    start_date_index = training_data_len 
    test_dates_full = df["date"].values[start_date_index:]
    
    # Trim dates and actuals to match predictions size
    start_index_trim = len(test_dates_full) - len(predictions_unscaled)
    dates_test = test_dates_full[start_index_trim:]
    y_test = Y_test_actual[start_index_trim:] # Y_test_actual is already trimmed in train_and_evaluate_lstm but we use the result here

    y_pred = predictions_unscaled

    # --- Plotting Logic ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Actual vs Predicted
    ax1 = axes[0]
    ax1.plot(dates_test, y_test, label='Actual', linewidth=2)
    ax1.plot(dates_test, y_pred, label='Predicted', linewidth=2, alpha=0.85)
    ax1.fill_between(dates_test, y_test, y_pred, alpha=0.25)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Gold Price (THB)', fontsize=12)
    ax1.set_title(f'Gold Price Prediction - {MODEL_NAME.upper()}', fontsize=14, fontweight='bold')
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
    plot_path = output_dir / f"predictions_{MODEL_NAME}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📈 Prediction plot saved to: {plot_path}")
    plt.show() 
    plt.close()

# -------------------- END PLOTTING LSTM FUNCTIONS -------------------- #
# เพิ่มฟังก์ชันนี้ต่อจาก plot_predictions (หรือ plot_lstm_predictions ถ้าคุณเพิ่มแล้ว)

def plot_model_comparison(prediction_results: dict, dates_test, output_dir: Path):
    """
    พล็อตเส้นทำนายของทุกโมเดลเทียบกับราคาจริงในกราฟเดียว
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        print("⚠️  Matplotlib not available, skipping comparison plot.")
        return

    print("📈 Generating ALL Model Comparison Plot...")

    # ดึงค่า Actual (ราคาจริง) ออกมา ซึ่งควรจะเป็นค่าเดียวกันสำหรับทุกโมเดล
    # ใช้ค่า Actual จากโมเดลแรกใน Dictionary
    model_names = list(prediction_results.keys())
    y_actual = prediction_results[model_names[0]]["y_actual"]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    
    # 1. Plot Actual Price
    ax.plot(dates_test[:len(y_actual)], y_actual, label='Actual Price', linewidth=3, color='black', alpha=0.9)

    # 2. Plot Predicted Prices for all models
    for name in model_names:
        y_pred = prediction_results[name]["y_pred"]
        # ต้องแน่ใจว่ามิติของ y_pred และ dates_test ตรงกันสำหรับ LSTM
        plot_len = min(len(y_actual), len(y_pred)) 
        
        ax.plot(dates_test[:plot_len], y_pred[:plot_len], label=f'Predicted - {name.upper()}', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Gold Price (THB)', fontsize=14)
    ax.set_title('Gold Price Prediction Comparison Across Models', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.4)
    
    # Format Date Axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f"predictions_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {plot_path}")
    plt.show() 
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

    # แยก LSTM placeholder model ออกไปก่อนเพื่อไม่ให้ train ด้วย Scikit-learn
    lstm_placeholder_model = models.pop("lstm", None)

    print(f"\n🎯 Available models: {', '.join(models.keys()).upper()}")

    # เทรนและประเมิน
    results, trained_models, splits = train_and_evaluate(
        models, X, y, test_size=args.test_size
    )

    # 2. --- LSTM MODEL TRAINING (If available) ---
    if HAS_TF and lstm_placeholder_model:
        # Ensure only rows that were valid in SKLearn split are used, to maintain test set integrity
        df_lstm = df.loc[X.index].copy() 

        lstm_results, lstm_trained_models, _ = train_and_evaluate_lstm(
            df_lstm, 
            feature_cols, 
            args.test_size
        )
        
        # Merge LSTM results with Sklearn results
        results.update(lstm_results)
        trained_models.update(lstm_trained_models)

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
        
        # --- NEW: Collect Predictions for Comparison Plot ---
        all_predictions = {}
        df_lstm_for_plot = df.loc[X.index].copy() # ใช้ข้อมูลที่ผ่านการ clean แล้ว
        
        print("\n" + "=" * 60)
        print("💾 Collecting Predictions for ALL Models...")

        for name, model in trained_models.items():
            y_actual, y_pred = get_predictions(
                model, 
                name, 
                X_test, 
                y_test, 
                df_lstm_for_plot, 
                feature_cols, 
                args.test_size
            )
            all_predictions[name] = {"y_actual": y_actual, "y_pred": y_pred}
            print(f"   Collected {name.upper()} predictions (Length: {len(y_pred)})")
        
        # --- Generate Comparison Plot ---
        plot_model_comparison(all_predictions, test_dates, args.results_dir)
        
        # --- Generate Individual Plots (Optional, ถ้าต้องการดูกราฟ Residuals) ---
        # ถ้าคุณต้องการให้แสดงกราฟ Residuals ของแต่ละโมเดลด้วย ให้เพิ่มโค้ดนี้
        # for name, model in trained_models.items():
        #     if name != "lstm":
        #         plot_predictions(model, X_test, y_test, test_dates, name, args.results_dir)
        #     # Note: ไม่จำเป็นต้องพล็อต LSTM แยกแล้ว เพราะข้อมูลไม่ครบ

    print("\n✅ Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
