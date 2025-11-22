#!/bin/bash
# install_updates.sh - สคริปต์ติดตั้งการอัพเดทระบบทำนายราคาทอง

echo "🚀 Gold Price Prediction System - Update Installer"
echo "=================================================="
echo ""

# ตรวจสอบ directory
if [ ! -d "data" ]; then
    echo "❌ Error: ไม่พบโฟลเดอร์ 'data'"
    echo "   กรุณารันสคริปต์นี้ที่ /Users/nichanun/Desktop/DSDN"
    exit 1
fi

echo "✅ พบโฟลเดอร์โปรเจกต์"
echo ""

# Backup ไฟล์เก่า
echo "📦 กำลัง backup ไฟล์เก่า..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

if [ -f "daily_pipeline.py" ]; then
    cp daily_pipeline.py "$BACKUP_DIR/"
    echo "   ✓ Backed up daily_pipeline.py"
fi

if [ -f "build_feature_store_btc.py" ]; then
    cp build_feature_store_btc.py "$BACKUP_DIR/"
    echo "   ✓ Backed up build_feature_store_btc.py"
fi

if [ -f "predict_gold.py" ]; then
    cp predict_gold.py "$BACKUP_DIR/"
    echo "   ✓ Backed up predict_gold.py"
fi

echo "   📁 Backup saved to: $BACKUP_DIR/"
echo ""

# แสดงรายการไฟล์ที่จะติดตั้ง
echo "📋 ไฟล์ที่จะติดตั้ง:"
echo "   1. gold_config.py (ใหม่) - ไฟล์ตั้งค่าระบบ"
echo "   2. predict_gold_skip_sundays.py (ใหม่) - สคริปต์ทำนายแบบใหม่"
echo "   3. daily_pipeline.py (อัพเดท) - แก้ไข bug และเพิ่ม feature"
echo "   4. build_feature_store_btc.py (อัพเดท) - รองรับการเลือกราคาทอง"
echo ""

# ยืนยันการติดตั้ง
read -p "ต้องการดำเนินการต่อหรือไม่? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ ยกเลิกการติดตั้ง"
    exit 0
fi

echo ""
echo "🔧 กำลังติดตั้ง..."
echo ""

# ตรวจสอบและคัดลอกไฟล์
copy_file_if_exists() {
    local source=$1
    local dest=$2
    local desc=$3
    
    if [ -f "$source" ]; then
        cp "$source" "$dest"
        echo "   ✅ $desc"
        return 0
    else
        echo "   ⚠️  ไม่พบ: $source"
        return 1
    fi
}

# ติดตั้งไฟล์ใหม่
copy_file_if_exists "gold_config.py" "." "ติดตั้ง gold_config.py"
copy_file_if_exists "predict_gold_skip_sundays.py" "." "ติดตั้ง predict_gold_skip_sundays.py"

echo ""
echo "✅ การติดตั้งเสร็จสมบูรณ์!"
echo ""

# แสดงขั้นตอนถัดไป
echo "📝 ขั้นตอนถัดไป:"
echo ""
echo "1. แก้ไขไฟล์ตั้งค่า:"
echo "   nano gold_config.py"
echo "   หรือ"
echo "   open -e gold_config.py"
echo ""
echo "2. เลือกประเภทราคาทอง (แนะนำ: gold_bar_sell):"
echo "   GOLD_PRICE_TYPE = \"gold_bar_sell\""
echo ""
echo "3. ทดสอบระบบ:"
echo "   python3 build_feature_store_btc.py"
echo "   python3 predict_gold_skip_sundays.py --days 1"
echo ""
echo "4. รัน pipeline:"
echo "   python3 daily_pipeline.py"
echo ""

# แสดง Quick Start Guide
if [ -f "QUICK_START.md" ]; then
    echo "📖 อ่านคู่มือเพิ่มเติม:"
    echo "   cat QUICK_START.md"
    echo "   หรือ"
    echo "   open QUICK_START.md"
    echo ""
fi

echo "🎉 ติดตั้งเรียบร้อย!"
echo "=================================================="
