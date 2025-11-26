#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
performance_analysis.py - Performance/Accuracy Trade-off Analysis
Fixed version: No emoji in matplotlib (avoids font warnings)
"""

import time
import psutil
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Matplotlib configuration (suppress emoji warnings)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use font without emoji

# ==================== CONFIG ====================
def find_project_root():
    current = Path.cwd()
    if (current / "data" / "Feature_store").exists():
        return current
    if (current.parent / "data" / "Feature_store").exists():
        return current.parent
    return current

PROJECT_ROOT = find_project_root()
MODEL_DIR = PROJECT_ROOT / "model"
FEATURE_STORE = PROJECT_ROOT / "data" / "Feature_store" / "feature_store.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

# ==================== FUNCTIONS ====================

def load_model_and_metadata():
    """Load model and metadata"""
    model_path = MODEL_DIR / "best_model.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
    
    return model, metadata

def load_test_data():
    """Load and split test data (last 20%)"""
    df = pd.read_csv(FEATURE_STORE, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Get features from metadata if available
    try:
        _, metadata = load_model_and_metadata()
        feature_cols = metadata['features']
    except:
        # Fallback: construct features manually
        base_vars = ['gold', 'fx', 'cpi', 'oil', 'set']
        if 'btc' in df.columns:
            base_vars.append('btc')
        
        feature_cols = []
        for var in base_vars:
            feature_cols.extend([
                f"{var}_lag1",
                f"{var}_lag3",
                f"{var}_roll7_mean",
                f"{var}_pct_change"
            ])
        feature_cols.extend(base_vars)
    
    # Split data (last 20% for test)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # Remove NaN
    valid_idx = ~(test_df[feature_cols].isna().any(axis=1) | test_df['gold_next'].isna())
    test_df = test_df[valid_idx]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['gold_next'].values
    
    return X_test, y_test, feature_cols

def measure_inference_time(model, X_test, n_iterations=1000):
    """Measure inference latency"""
    latencies = []
    
    # Warm-up
    for _ in range(10):
        _ = model.predict(X_test[:1])
    
    # Measure
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X_test[:1])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    return {
        'mean_ms': np.mean(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'std_ms': np.std(latencies)
    }

def measure_throughput(model, X_test, duration_seconds=5):
    """Measure throughput (predictions/sec)"""
    count = 0
    start = time.perf_counter()
    end_time = start + duration_seconds
    
    while time.perf_counter() < end_time:
        _ = model.predict(X_test[:1])
        count += 1
    
    elapsed = time.perf_counter() - start
    return count / elapsed

def measure_memory_usage(model):
    """Measure memory usage"""
    import sys
    
    # Model size
    model_size_bytes = sys.getsizeof(joblib.dump(model, '/tmp/temp_model.pkl'))
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Process memory
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)
    
    return {
        'model_size_mb': model_size_mb,
        'process_memory_mb': memory_mb
    }

def measure_model_accuracy(model, X_test, y_test):
    """Measure model accuracy"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy percentage (1 - MAPE)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    accuracy = 100 - mape
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'accuracy_pct': accuracy
    }

def analyze_batch_size_tradeoff(model, X_test, batch_sizes=[1, 10, 50, 100]):
    """Analyze performance at different batch sizes"""
    results = []
    
    for batch_size in batch_sizes:
        # Prepare batch
        batch = X_test[:batch_size]
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model.predict(batch)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        total_latency = np.mean(latencies)
        per_sample_latency = total_latency / batch_size
        p95_latency = np.percentile(latencies, 95)
        
        # Measure throughput
        count = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:  # 1 second
            _ = model.predict(batch)
            count += batch_size
        throughput = count / (time.perf_counter() - start)
        
        results.append({
            'batch_size': batch_size,
            'total_latency_ms': total_latency,
            'latency_per_sample_ms': per_sample_latency,
            'p95_latency_ms': p95_latency,
            'throughput_samples_per_sec': throughput
        })
    
    return pd.DataFrame(results)

def plot_performance_tradeoff(latency_data, accuracy_data, batch_data, memory_data, output_path):
    """Create performance trade-off visualization (NO EMOJI)"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Latency Distribution
    ax1 = axes[0, 0]
    latencies = [latency_data['mean_ms'], latency_data['p50_ms'], 
                 latency_data['p95_ms'], latency_data['p99_ms']]
    labels = ['Mean', 'P50', 'P95', 'P99']
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    bars = ax1.bar(labels, latencies, color=colors, alpha=0.8)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('[BAR CHART] Inference Latency Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}ms',
                ha='center', va='bottom', fontsize=10)
    
    # 2. Batch Size Impact
    ax2 = axes[0, 1]
    ax2.plot(batch_data['batch_size'], batch_data['latency_per_sample_ms'], 
             marker='o', linewidth=2, markersize=8, color='#3498db', label='Latency/Sample')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Latency per Sample (ms)', fontsize=12, color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.set_title('[DIRECT HIT] Batch Size vs Latency', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Twin axis for throughput
    ax2_twin = ax2.twinx()
    ax2_twin.plot(batch_data['batch_size'], batch_data['throughput_samples_per_sec'],
                  marker='s', linewidth=2, markersize=8, color='#e74c3c', label='Throughput')
    ax2_twin.set_ylabel('Throughput (samples/sec)', fontsize=12, color='#e74c3c')
    ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax2_twin.legend(loc='upper right')
    
    # 3. Accuracy vs Latency Scatter
    ax3 = axes[1, 0]
    ax3.scatter([latency_data['mean_ms']], [accuracy_data['accuracy_pct']], 
                s=500, alpha=0.6, c='#2ecc71', edgecolors='black', linewidth=2)
    ax3.set_xlabel('Mean Latency (ms)', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('[ROCKET] Accuracy vs Latency Trade-off', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.set_ylim([95, 100])
    
    # Add annotation
    ax3.annotate(f'Current Model\nAcc: {accuracy_data["accuracy_pct"]:.2f}%\nLatency: {latency_data["mean_ms"]:.2f}ms',
                xy=(latency_data['mean_ms'], accuracy_data['accuracy_pct']),
                xytext=(latency_data['mean_ms'] * 1.5, accuracy_data['accuracy_pct'] - 1),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    # 4. Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    [LIGHT BULB] PERFORMANCE SUMMARY
    
    ACCURACY METRICS:
    - MAE: {accuracy_data['mae']:.2f} THB
    - RMSE: {accuracy_data['rmse']:.2f} THB
    - R² Score: {accuracy_data['r2']:.4f}
    - Accuracy: {accuracy_data['accuracy_pct']:.2f}%
    
    LATENCY METRICS:
    - Mean: {latency_data['mean_ms']:.2f} ms
    - P95: {latency_data['p95_ms']:.2f} ms
    - P99: {latency_data['p99_ms']:.2f} ms
    
    THROUGHPUT:
    - {memory_data['throughput']:.0f} predictions/sec
    
    MEMORY:
    - Model Size: {memory_data['model_size_mb']:.2f} MB
    - Process Memory: {memory_data['process_memory_mb']:.2f} MB
    
    TRADE-OFF SCORE:
    - {accuracy_data['accuracy_pct'] / latency_data['mean_ms']:.2f}
      (Higher is better)
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("⚡ PERFORMANCE/ACCURACY TRADE-OFF ANALYSIS")
    print("=" * 70)
    
    # Load model
    print("\n📦 Loading model...")
    model, metadata = load_model_and_metadata()
    model_type = metadata.get('model_type', 'unknown').upper()
    print(f"✅ Loaded {model_type} model")
    
    # Load test data
    print("\n📊 Loading test data...")
    X_test, y_test, features = load_test_data()
    print(f"   Test samples: {len(X_test)}")
    
    # Measure accuracy
    print("\n🎯 Measuring accuracy...")
    accuracy_metrics = measure_model_accuracy(model, X_test, y_test)
    print(f"   MAE: {accuracy_metrics['mae']:.2f} THB")
    print(f"   RMSE: {accuracy_metrics['rmse']:.2f} THB")
    print(f"   Accuracy: {accuracy_metrics['accuracy_pct']:.2f}%")
    
    # Measure performance
    print("\n⚡ Measuring performance...")
    latency_metrics = measure_inference_time(model, X_test, n_iterations=1000)
    print(f"   Mean Latency: {latency_metrics['mean_ms']:.2f} ms")
    print(f"   P95 Latency: {latency_metrics['p95_ms']:.2f} ms")
    print(f"   P99 Latency: {latency_metrics['p99_ms']:.2f} ms")
    
    throughput = measure_throughput(model, X_test, duration_seconds=5)
    print(f"   Throughput: {throughput:.0f} predictions/sec")
    
    memory_metrics = measure_memory_usage(model)
    print(f"   Model Size: {memory_metrics['model_size_mb']:.2f} MB")
    print(f"   Process Memory: {memory_metrics['process_memory_mb']:.2f} MB")
    
    # Batch analysis
    print("\n📦 Analyzing batch size impact...")
    batch_results = analyze_batch_size_tradeoff(model, X_test)
    print(batch_results.to_string(index=False))
    
    # Create visualization
    print("\n📈 Generating visualizations...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = RESULTS_DIR / f"performance_tradeoff_{timestamp}.png"
    
    memory_metrics['throughput'] = throughput
    plot_performance_tradeoff(latency_metrics, accuracy_metrics, batch_results, 
                             memory_metrics, plot_path)
    print(f"📊 Trade-off plot saved to: {plot_path}")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'accuracy': accuracy_metrics,
        'latency': latency_metrics,
        'throughput': throughput,
        'memory': memory_metrics,
        'batch_analysis': batch_results.to_dict('records'),
        'trade_off_score': accuracy_metrics['accuracy_pct'] / latency_metrics['mean_ms']
    }
    
    report_path = RESULTS_DIR / f"performance_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRADE-OFF SUMMARY")
    print("=" * 70)
    print(f"✅ Model achieves {accuracy_metrics['accuracy_pct']:.2f}% accuracy")
    print(f"⚡ With latency of {latency_metrics['mean_ms']:.2f} ms (P95: {latency_metrics['p95_ms']:.2f} ms)")
    print(f"🚀 Throughput: {throughput:.0f} predictions/sec")
    print(f"💾 Memory footprint: {memory_metrics['model_size_mb']:.2f} MB")
    print(f"🎯 Trade-off Score: {report['trade_off_score']:.2f}")
    print(f"\n💡 Recommendations:")
    
    if report['trade_off_score'] > 300:
        print("   ✅ Excellent balance between speed and accuracy!")
    elif report['trade_off_score'] > 100:
        print("   ⚠️  Good balance, but consider optimization")
    else:
        print("   ❌ Poor trade-off - consider simpler model")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()