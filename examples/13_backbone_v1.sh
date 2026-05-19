#!/bin/bash
# Example 13: Multi-Head MobileNetV1 backbone
# Demonstrates --backbone v1, which produces an mnv1_-prefixed output.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 13: Multi-Head MobileNetV1 (--backbone v1)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/backbone_v1
OUT_NAME=mnv1_multi

python ../src/create_quantized_mobilenet.py \
    --backbone v1 \
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
echo "Example 13 complete."
