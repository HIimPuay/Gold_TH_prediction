#!/usr/bin/env python3
"""
dashboard.py - แดชบอร์ดวิเคราะห์และสรุปผล
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/Users/nichanun/Desktop/DSDN")
FEATURE_STORE = PROJECT_ROOT / "data" / "Feature_store" / "feature_store.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

def load_data():
    """โหลดข้อมูล"""
    if not FEATURE_STORE.exists():
        raise FileNotFoundError(f"Feature store not found: {FEATURE_STORE}")
    
    df = pd.read_csv(FEATURE_STORE, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)

def calculate_metrics(df, days=30):
    """คำนวณเมทริกส์สำคัญ"""
    recent = df.tail(days)
    
    metrics = {
        'current_price': recent.iloc[-1]['gold'],
        'min_30d': recent['gold'].min(),
        'max_30d': recent['gold'].max(),
        'avg_30d': recent['gold'].mean(),
        'std_30d': recent['gold'].std(),
        'change_30d': recent.iloc[-1]['gold'] - recent.iloc[0]['gold'],
        'change_30d_pct': ((recent.iloc[-1]['gold'] - recent.iloc[0]['gold']) / recent.iloc[0]['gold']) * 100,
        'volatility': recent['gold'].std() / recent['gold'].mean() * 100,
    }
    
    return metrics

def analyze_correlations(df, days=180):
    """วิเคราะห์ความสัมพันธ์"""
    recent = df.tail(days)
    
    # คอลัมน์ที่ต้องการวิเคราะห์
    cols = ['gold', 'fx', 'cpi', 'oil', 'set']
    if 'btc' in recent.columns:
        cols.append('btc')
    
    corr_matrix = recent[cols].corr()['gold'].sort_values(ascending=False)
    
    return corr_matrix

def detect_trend(df, window=7):
    """ตรวจหาเทรนด์ (ขึ้น/ลง/คงที่)"""
    recent = df.tail(window)
    
    # Simple linear regression slope
    x = np.arange(len(recent))
    y = recent['gold'].values
    slope = np.polyfit(x, y, 1)[0]
    
    # คำนวณ % change เฉลี่ย
    avg_change = recent['gold'].pct_change().mean() * 100
    
    if abs(avg_change) < 0.1:
        trend = "คงที่"
        emoji = "➡️"
    elif avg_change > 0:
        trend = "ขาขึ้น"
        emoji = "📈"
    else:
        trend = "ขาลง"
        emoji = "📉"
    
    return {
        'trend': trend,
        'emoji': emoji,
        'slope': slope,
        'avg_change_pct': avg_change
    }

def load_latest_prediction():
    """โหลดการทำนายล่าสุด"""
    pred_files = list(RESULTS_DIR.glob("predictions_7days_*.csv"))
    
    if not pred_files:
        return None
    
    # เลือกไฟล์ล่าสุด
    latest_file = max(pred_files, key=lambda p: p.stat().st_mtime)
    df_pred = pd.read_csv(latest_file, parse_dates=['date'])
    
    return df_pred

def print_dashboard():
    """แสดงแดชบอร์ด"""
    print("\n" + "=" * 70)
    print("🏆 GOLD PRICE ANALYSIS DASHBOARD")
    print("=" * 70)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # โหลดข้อมูล
    df = load_data()
    
    # ข้อมูลล่าสุด
    latest = df.iloc[-1]
    print(f"\n📊 CURRENT STATUS")
    print("-" * 70)
    print(f"Latest Date:     {latest['date'].strftime('%Y-%m-%d')}")
    print(f"Gold Price:      {latest['gold']:,.2f} THB")
    print(f"USD/THB:         {latest['fx']:.4f}")
    print(f"CPI:             {latest['cpi']:.2f}")
    print(f"Oil Price:       ${latest['oil']:.2f}")
    print(f"SET Index:       {latest['set']:,.2f}")
    if 'btc' in latest.index and not pd.isna(latest['btc']):
        print(f"Bitcoin:         {latest['btc']:,.2f} THB")
    
    # เมทริกส์ 30 วัน
    metrics = calculate_metrics(df, days=30)
    print(f"\n📈 30-DAY STATISTICS")
    print("-" * 70)
    print(f"Current:         {metrics['current_price']:,.2f} THB")
    print(f"Range:           {metrics['min_30d']:,.2f} - {metrics['max_30d']:,.2f} THB")
    print(f"Average:         {metrics['avg_30d']:,.2f} THB (±{metrics['std_30d']:.2f})")
    print(f"Change (30d):    {'+' if metrics['change_30d'] > 0 else ''}{metrics['change_30d']:,.2f} THB ({metrics['change_30d_pct']:+.2f}%)")
    print(f"Volatility:      {metrics['volatility']:.2f}%")
    
    # เทรนด์
    trend = detect_trend(df, window=7)
    print(f"\n{trend['emoji']} 7-DAY TREND")
    print("-" * 70)
    print(f"Direction:       {trend['trend']}")
    print(f"Avg Daily Δ:     {trend['avg_change_pct']:+.3f}%")
    
    # Correlation
    print(f"\n🔗 CORRELATION WITH GOLD (180 days)")
    print("-" * 70)
    corr = analyze_correlations(df, days=180)
    for var, value in corr.items():
        if var != 'gold':
            bar_length = int(abs(value) * 20)
            bar = "█" * bar_length
            print(f"{var.upper():8s} {value:+.3f}  {bar}")
    
    # การทำนาย
    df_pred = load_latest_prediction()
    if df_pred is not None:
        print(f"\n🔮 7-DAY FORECAST")
        print("-" * 70)
        for _, row in df_pred.head(7).iterrows():
            emoji = "📈" if row['change'] > 0 else "📉" if row['change'] < 0 else "➡️"
            print(f"{row['date'].strftime('%Y-%m-%d')}  {emoji}  {row['predicted_price']:,.2f} THB  "
                  f"({'+' if row['change'] > 0 else ''}{row['change_pct']:.2f}%)")
    else:
        print(f"\n⚠️  No predictions available (run predict_gold.py)")
    
    # คำแนะนำ
    print(f"\n💡 TRADING SIGNALS")
    print("-" * 70)
    
    # Signal 1: เทรนด์ระยะสั้น
    if trend['trend'] == "ขาขึ้น":
        print("🟢 Short-term: แนวโน้มขาขึ้น - พิจารณาซื้อ")
    elif trend['trend'] == "ขาลง":
        print("🔴 Short-term: แนวโน้มขาลง - พิจารณาขาย/รอ")
    else:
        print("🟡 Short-term: แนวโน้มคงที่ - รอสัญญาณ")
    
    # Signal 2: Volatility
    if metrics['volatility'] > 1.5:
        print("⚠️  Volatility: สูง - ควรระวังความเสี่ยง")
    elif metrics['volatility'] < 0.5:
        print("✅ Volatility: ต่ำ - ตลาดค่อนข้างเสถียร")
    else:
        print("➡️  Volatility: ปานกลาง")
    
    # Signal 3: การทำนาย
    if df_pred is not None and len(df_pred) > 0:
        avg_pred_change = df_pred['change_pct'].mean()
        if avg_pred_change > 0.5:
            print("📈 7-day Outlook: คาดว่าจะขึ้น - โอกาสดีในการซื้อ")
        elif avg_pred_change < -0.5:
            print("📉 7-day Outlook: คาดว่าจะลง - พิจารณารอหรือขาย")
        else:
            print("➡️  7-day Outlook: คาดว่าจะคงที่")
    
    print("\n" + "=" * 70)
    print("⚠️  Disclaimer: ข้อมูลนี้ใช้สำหรับการศึกษาเท่านั้น")
    print("   ไม่ควรใช้เป็นคำแนะนำในการลงทุนโดยตรง")
    print("=" * 70 + "\n")

def export_summary():
    """ส่งออกสรุปเป็น CSV"""
    df = load_data()
    metrics = calculate_metrics(df, days=30)
    trend = detect_trend(df, window=7)
    
    summary = {
        'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'latest_date': df.iloc[-1]['date'].strftime('%Y-%m-%d'),
        'current_price': metrics['current_price'],
        'change_30d': metrics['change_30d'],
        'change_30d_pct': metrics['change_30d_pct'],
        'volatility': metrics['volatility'],
        'trend_7d': trend['trend'],
        'avg_change_7d_pct': trend['avg_change_pct']
    }
    
    df_summary = pd.DataFrame([summary])
    output_path = RESULTS_DIR / f"summary_{datetime.now().strftime('%Y%m%d')}.csv"
    df_summary.to_csv(output_path, index=False)
    
    print(f"💾 Summary exported to: {output_path}")

def main():
    try:
        print_dashboard()
        export_summary()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()