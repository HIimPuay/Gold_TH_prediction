#!/usr/bin/env python3
"""
fix_feature_store_columns.py - แก้ไขชื่อคอลัมน์ใน feature store

ปัญหา: โมเดลคาดหวัง:
- gold_roll7_mean แต่ได้ gold_roll7
- gold_pct_change แต่ได้ gold_pct

แก้ไข: เปลี่ยนชื่อคอลัมน์ให้ตรงกับที่โมเดลต้องการ
"""

import pandas as pd
from pathlib import Path
import sys

def fix_feature_store(input_path: Path, output_path: Path = None):
    """แก้ไขชื่อคอลัมน์ใน feature store"""
    
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False
    
    if output_path is None:
        output_path = input_path
    
    print(f"📖 Reading: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"   Original columns: {len(df.columns)}")
    print(f"   Original rows: {len(df)}")
    
    # รายการคอลัมน์ที่ต้องเปลี่ยนชื่อ
    rename_map = {}
    
    # สำหรับทุกตัวแปร
    variables = ['gold', 'fx', 'cpi', 'oil', 'set', 'btc']
    
    for var in variables:
        # _roll7 → _roll7_mean
        old_roll = f"{var}_roll7"
        new_roll = f"{var}_roll7_mean"
        if old_roll in df.columns and new_roll not in df.columns:
            rename_map[old_roll] = new_roll
        
        # _pct → _pct_change
        old_pct = f"{var}_pct"
        new_pct = f"{var}_pct_change"
        if old_pct in df.columns and new_pct not in df.columns:
            rename_map[old_pct] = new_pct
    
    if not rename_map:
        print("✅ No columns to rename - already correct!")
        return True
    
    print(f"\n🔧 Renaming columns:")
    for old, new in rename_map.items():
        print(f"   {old:20s} → {new}")
    
    # เปลี่ยนชื่อ
    df = df.rename(columns=rename_map)
    
    # บันทึก
    print(f"\n💾 Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print(f"✅ Done!")
    print(f"   New columns: {len(df.columns)}")
    print(f"   Rows: {len(df)}")
    
    # แสดงคอลัมน์ทั้งหมด
    print(f"\n📋 All columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix feature store column names"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Feature_store/feature_store.csv"),
        help="Input feature store path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before fixing"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔧 Fix Feature Store Column Names")
    print("=" * 70)
    print()
    
    # Backup
    if args.backup:
        from datetime import datetime
        backup_path = args.input.parent / f"{args.input.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"📦 Creating backup: {backup_path}")
        import shutil
        shutil.copy(args.input, backup_path)
        print(f"✅ Backup created\n")
    
    # Fix
    success = fix_feature_store(args.input, args.output)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ Feature store fixed successfully!")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("❌ Failed to fix feature store")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
