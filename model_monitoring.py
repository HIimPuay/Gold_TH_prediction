#!/usr/bin/env python3
"""
model_monitoring.py - ระบบตรวจสอบ Model Performance และ Concept Drift
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MONITORING_FILE = RESULTS_DIR / "model_monitoring.json"

# ==================== METRICS ====================

def calculate_moving_average_accuracy(predictions_df, window_sizes=[7, 30]):
    """
    คำนวณ Moving Average Accuracy (MAcc)
    
    MAcc(t, w) = (1/w) * Σ(Acc(t-k)) for k=0 to w-1
    
    Args:
        predictions_df: DataFrame with columns ['date', 'actual', 'predicted']
        window_sizes: list of window sizes (e.g., [7, 30])
    
    Returns:
        DataFrame with moving average accuracy
    """
    df = predictions_df.copy()
    
    # คำนวณ accuracy แต่ละวัน (1 - MAPE)
    df['error_pct'] = np.abs((df['actual'] - df['predicted']) / df['actual']) * 100
    df['accuracy'] = 100 - df['error_pct']
    
    results = {'date': df['date']}
    
    for w in window_sizes:
        col_name = f'MAcc_{w}d'
        results[col_name] = df['accuracy'].rolling(window=w, min_periods=1).mean()
    
    return pd.DataFrame(results)


def calculate_accuracy_decay_rate(macc_df, window_col='MAcc_7d'):
    """
    คำนวณ Accuracy Decay Rate
    
    DecayRate = (Acc(t1) - Acc(t2)) / (t2 - t1)
    
    Args:
        macc_df: DataFrame from calculate_moving_average_accuracy
        window_col: column name to calculate decay rate
    
    Returns:
        float: decay rate per day
    """
    if len(macc_df) < 2:
        return 0.0
    
    # ใช้ค่า 2 จุดล่าสุด
    t1_acc = macc_df[window_col].iloc[-2]
    t2_acc = macc_df[window_col].iloc[-1]
    
    decay_rate = (t1_acc - t2_acc)  # per day
    
    return decay_rate


def detect_concept_drift(macc_df, threshold=-0.5, window='MAcc_7d'):
    """
    ตรวจจับ Concept Drift
    
    Drift detected if:
    - DecayRate < threshold (accuracy กำลังลดลงเร็ว)
    - Accuracy ลดลงต่อเนื่อง 3 วันขึ้นไป
    
    Returns:
        dict with drift detection results
    """
    decay_rate = calculate_accuracy_decay_rate(macc_df, window)
    
    # เช็คว่า accuracy ลดลงต่อเนื่อง
    recent_acc = macc_df[window].tail(5).values
    is_declining = all(recent_acc[i] > recent_acc[i+1] for i in range(len(recent_acc)-1))
    
    drift_detected = decay_rate < threshold or is_declining
    
    return {
        'drift_detected': drift_detected,
        'decay_rate': decay_rate,
        'current_accuracy': macc_df[window].iloc[-1],
        'is_declining': is_declining,
        'severity': 'HIGH' if decay_rate < -1.0 else 'MEDIUM' if decay_rate < threshold else 'LOW'
    }


def calculate_balance_index(predictions_df):
    """
    คำนวณ Balance Index - วัดว่าโมเดล bias ไปทางไหน
    
    BI = (จำนวนที่ predict สูงกว่าจริง - จำนวนที่ predict ต่ำกว่าจริง) / total
    
    BI = 0  → balanced
    BI > 0  → tends to over-predict
    BI < 0  → tends to under-predict
    
    Returns:
        float: balance index between -1 and 1
    """
    df = predictions_df.copy()
    
    over_predict = (df['predicted'] > df['actual']).sum()
    under_predict = (df['predicted'] < df['actual']).sum()
    total = len(df)
    
    balance_index = (over_predict - under_predict) / total
    
    return balance_index


def calculate_adaptation_speed(macc_df, window='MAcc_7d'):
    """
    คำนวณ Adaptation Speed - เร็วแค่ไหนที่โมเดลปรับตัว
    
    Based on: อัตราการเปลี่ยนแปลงของ accuracy
    
    Returns:
        dict with adaptation metrics
    """
    if len(macc_df) < 10:
        return {'speed': 'N/A', 'status': 'insufficient_data'}
    
    # คำนวณ variance ของ accuracy
    recent_acc = macc_df[window].tail(10)
    variance = recent_acc.var()
    
    # คำนวณ trend
    x = np.arange(len(recent_acc))
    slope = np.polyfit(x, recent_acc, 1)[0]
    
    # Speed classification
    if variance < 0.1:
        speed = 'STABLE'
    elif variance < 0.5:
        speed = 'MODERATE'
    else:
        speed = 'VOLATILE'
    
    return {
        'speed': speed,
        'variance': variance,
        'trend': 'IMPROVING' if slope > 0 else 'DECLINING' if slope < 0 else 'STABLE',
        'slope': slope
    }


# ==================== MONITORING REPORT ====================

def generate_monitoring_report(predictions_df):
    """
    สร้างรายงาน monitoring ครบถ้วน
    
    Args:
        predictions_df: DataFrame with ['date', 'actual', 'predicted']
    
    Returns:
        dict with complete monitoring report
    """
    # 1. Moving Average Accuracy
    macc_df = calculate_moving_average_accuracy(predictions_df, window_sizes=[7, 30])
    
    # 2. Drift Detection
    drift_7d = detect_concept_drift(macc_df, window='MAcc_7d')
    drift_30d = detect_concept_drift(macc_df, window='MAcc_30d')
    
    # 3. Balance Index
    balance_idx = calculate_balance_index(predictions_df)
    
    # 4. Adaptation Speed
    adaptation = calculate_adaptation_speed(macc_df)
    
    # 5. Overall Metrics
    latest_acc_7d = macc_df['MAcc_7d'].iloc[-1]
    latest_acc_30d = macc_df['MAcc_30d'].iloc[-1]
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'data_points': len(predictions_df),
        'date_range': {
            'start': predictions_df['date'].min(),
            'end': predictions_df['date'].max()
        },
        'accuracy': {
            'current_7d': latest_acc_7d,
            'current_30d': latest_acc_30d,
            'decay_rate_7d': drift_7d['decay_rate'],
            'decay_rate_30d': drift_30d['decay_rate']
        },
        'drift_detection': {
            '7d': drift_7d,
            '30d': drift_30d
        },
        'balance_index': {
            'value': balance_idx,
            'interpretation': 'OVER-PREDICT' if balance_idx > 0.1 else 'UNDER-PREDICT' if balance_idx < -0.1 else 'BALANCED'
        },
        'adaptation': adaptation,
        'recommendations': []
    }
    
    # 6. Recommendations
    if drift_7d['drift_detected']:
        report['recommendations'].append('⚠️  Concept drift detected (7d). Consider retraining the model.')
    
    if latest_acc_7d < 95:
        report['recommendations'].append('📉 Accuracy below 95%. Investigate feature quality.')
    
    if abs(balance_idx) > 0.2:
        report['recommendations'].append(f'⚖️  Model is biased ({report["balance_index"]["interpretation"]}). Check training data balance.')
    
    if adaptation['speed'] == 'VOLATILE':
        report['recommendations'].append('📊 High volatility detected. Monitor closely for next few days.')
    
    if not report['recommendations']:
        report['recommendations'].append('✅ Model performance is stable.')
    
    return report, macc_df


def print_monitoring_report(report):
    """แสดงรายงานแบบสวยงาม"""
    print("\n" + "=" * 70)
    print("📊 MODEL PERFORMANCE MONITORING REPORT")
    print("=" * 70)
    print(f"📅 Generated: {report['generated_at'][:19]}")
    print(f"📈 Data Points: {report['data_points']}")
    print(f"🗓️  Date Range: {report['date_range']['start']} to {report['date_range']['end']}")
    
    print(f"\n🎯 ACCURACY METRICS")
    print("-" * 70)
    print(f"Current (7-day):   {report['accuracy']['current_7d']:.2f}%")
    print(f"Current (30-day):  {report['accuracy']['current_30d']:.2f}%")
    print(f"Decay Rate (7d):   {report['accuracy']['decay_rate_7d']:+.3f}% per day")
    print(f"Decay Rate (30d):  {report['accuracy']['decay_rate_30d']:+.3f}% per day")
    
    print(f"\n🔍 CONCEPT DRIFT DETECTION")
    print("-" * 70)
    drift_7d = report['drift_detection']['7d']
    drift_30d = report['drift_detection']['30d']
    print(f"7-day window:   {'🚨 DRIFT DETECTED' if drift_7d['drift_detected'] else '✅ NO DRIFT'} (Severity: {drift_7d['severity']})")
    print(f"30-day window:  {'🚨 DRIFT DETECTED' if drift_30d['drift_detected'] else '✅ NO DRIFT'} (Severity: {drift_30d['severity']})")
    
    print(f"\n⚖️  BALANCE INDEX")
    print("-" * 70)
    bi = report['balance_index']
    print(f"Value: {bi['value']:+.3f}")
    print(f"Interpretation: {bi['interpretation']}")
    
    print(f"\n🏃 ADAPTATION METRICS")
    print("-" * 70)
    adaptation = report['adaptation']
    print(f"Speed:    {adaptation['speed']}")
    print(f"Trend:    {adaptation['trend']}")
    print(f"Variance: {adaptation['variance']:.4f}")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 70)
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    print("\n" + "=" * 70 + "\n")


# ==================== SAVE & LOAD ====================

def save_monitoring_history(report):
    """บันทึกประวัติ monitoring"""
    MONITORING_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # โหลดประวัติเก่า
    history = []
    if MONITORING_FILE.exists():
        with open(MONITORING_FILE, 'r') as f:
            history = json.load(f)
    
    # เพิ่มรายงานใหม่
    history.append(report)
    
    # เก็บแค่ 100 รายงานล่าสุด
    history = history[-100:]
    
    # บันทึก
    with open(MONITORING_FILE, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"💾 Monitoring history saved to: {MONITORING_FILE}")


# ==================== MAIN ====================

def main():
    """ตัวอย่างการใช้งาน"""
    
    # สร้างข้อมูลทดสอบ (ในการใช้งานจริงให้โหลดจากไฟล์ predictions)
    print("📊 Creating sample data for demonstration...")
    
    dates = pd.date_range(start='2025-10-01', end='2025-11-19', freq='D')
    np.random.seed(42)
    
    # สร้างข้อมูล actual และ predicted
    actual = 60000 + np.cumsum(np.random.randn(len(dates)) * 100)
    predicted = actual + np.random.randn(len(dates)) * 300
    
    predictions_df = pd.DataFrame({
        'date': dates,
        'actual': actual,
        'predicted': predicted
    })
    
    # Generate report
    report, macc_df = generate_monitoring_report(predictions_df)
    
    # Print report
    print_monitoring_report(report)
    
    # Save history
    save_monitoring_history(report)
    
    # Export MAcc to CSV
    macc_path = RESULTS_DIR / f"moving_avg_accuracy_{datetime.now().strftime('%Y%m%d')}.csv"
    macc_df.to_csv(macc_path, index=False)
    print(f"📈 Moving Average Accuracy saved to: {macc_path}")


if __name__ == "__main__":
    main()
