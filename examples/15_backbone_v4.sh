#!/bin/bash
# Example 15: Multi-Head MobileNetV4 backbone
# Demonstrates --backbone v4, which builds the custom UIB-based
# MobileNetV4-Conv-S feature extractor (no Keras applications V4 exists).

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 15: Multi-Head MobileNetV4 (--backbone v4)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/backbone_v4
OUT_NAME=mnv4_multi

python ../src/create_quantized_mobilenet.py \
    --backbone v4 \
    --alpha 0.25 \
    --input-shape "96x96x1" \
    --heads "5,2,3" \
    --head-names "object_class,person_detection,age_group" \
    --output-dir "$OUT_DIR" \
    --output-name "$OUT_NAME" \
    --no-save-keras

echo ""
echo "Verifying TFLite output structure..."
python _verify_tflite.py "$OUT_DIR/${OUT_NAME}_int8.tflite" --expected-heads 3

echo ""
echo "Example 15 complete."
