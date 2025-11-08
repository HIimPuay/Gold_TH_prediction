#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_features.py - วิเคราะห์ความสำคัญของ features
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
RESULTS_DIR = PROJECT_ROOT / "results"

def load_model_and_metadata():
    """โหลดโมเดลและ metadata"""
    model_path = MODEL_DIR / "best_model.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found. Run train_model.py first!")
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    return model, metadata

def get_feature_importance(model, feature_names, model_type):
    """ดึง feature importance จากโมเดล"""
    
    # Linear models - ใช้ coefficients
    if model_type in ['linear', 'ridge', 'lasso']:
        importance = np.abs(model.coef_)
        
    # Tree-based models
    elif model_type in ['rf', 'gbm', 'xgb', 'lgb']:
        importance = model.feature_importances_
        
    else:
        print(f"⚠️  Model type '{model_type}' not supported for feature importance")
        return None
    
    # สร้าง DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # เรียงลำดับ
    df_importance = df_importance.sort_values('importance', ascending=False)
    
    # Normalize เป็นเปอร์เซ็นต์
    df_importance['importance_pct'] = (
        df_importance['importance'] / df_importance['importance'].sum() * 100
    )
    
    return df_importance

def plot_feature_importance(df_importance, model_type, top_n=20):
    """พล็อต feature importance"""
    
    # เลือก top N features
    df_plot = df_importance.head(top_n)
    
    # สร้างกราฟ
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(df_plot['feature'], df_plot['importance_pct'], 
                    color='steelblue', alpha=0.8)
    
    # เพิ่มค่าที่ปลายแท่ง
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.text(row['importance_pct'], i, f" {row['importance_pct']:.1f}%", 
                va='center', fontsize=9)
    
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - {model_type.upper()}', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # บันทึก
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / f"feature_importance_{model_type}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved to: {plot_path}")
    plt.close()

def analyze_feature_groups(df_importance):
    """วิเคราะห์ตามกลุ่มตัวแปร"""
    
    # แบ่งกลุ่ม
    groups = {
        'gold': [],
        'fx': [],
        'cpi': [],
        'oil': [],
        'set': [],
        'btc': []
    }
    
    for _, row in df_importance.iterrows():
        feature = row['feature']
        for key in groups.keys():
            if feature.startswith(key):
                groups[key].append(row['importance_pct'])
                break
    
    # คำนวณผลรวม
    group_importance = {
        k: sum(v) for k, v in groups.items() if v
    }
    
    # เรียงลำดับ
    group_importance = dict(
        sorted(group_importance.items(), key=lambda x: x[1], reverse=True)
    )
    
    return group_importance

def print_summary(df_importance, group_importance, model_type):
    """แสดงสรุปผล"""
    
    print("\n" + "=" * 70)
    print(f"📊 FEATURE IMPORTANCE ANALYSIS - {model_type.upper()}")
    print("=" * 70)
    
    print(f"\n🔝 Top 10 Most Important Features:")
    print("-" * 70)
    for i, (_, row) in enumerate(df_importance.head(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:30s} {row['importance_pct']:6.2f}%")
    
    print(f"\n📦 Feature Groups Importance:")
    print("-" * 70)
    for var, importance in group_importance.items():
        bar_length = int(importance / 2)  # Scale for display
        bar = "█" * bar_length
        print(f"{var.upper():6s} {bar:50s} {importance:6.2f}%")
    
    print("\n" + "=" * 70)
    
    # Insights
    top_group = max(group_importance.items(), key=lambda x: x[1])
    print(f"\n💡 Insights:")
    print(f"   • Most important group: {top_group[0].upper()} ({top_group[1]:.1f}%)")
    print(f"   • Total features: {len(df_importance)}")
    print(f"   • Top 10 features account for: {df_importance.head(10)['importance_pct'].sum():.1f}%")
    print("=" * 70 + "\n")

def main():
    try:
        print("🚀 Starting Feature Importance Analysis...")
        
        # โหลดโมเดล
        print("📦 Loading model...")
        model, metadata = load_model_and_metadata()
        
        model_type = metadata['model_type']
        feature_names = metadata['features']
        
        print(f"✅ Loaded {model_type.upper()} model with {len(feature_names)} features")
        
        # วิเคราะห์ feature importance
        print("\n🔍 Analyzing feature importance...")
        df_importance = get_feature_importance(model, feature_names, model_type)
        
        if df_importance is None:
            return
        
        # วิเคราะห์ตามกลุ่ม
        group_importance = analyze_feature_groups(df_importance)
        
        # แสดงผล
        print_summary(df_importance, group_importance, model_type)
        
        # บันทึก CSV
        csv_path = RESULTS_DIR / f"feature_importance_{model_type}.csv"
        df_importance.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")
        
        # สร้างกราฟ
        print("\n📊 Creating visualization...")
        plot_feature_importance(df_importance, model_type)
        
        print("\n✅ Analysis complete!")
        
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()