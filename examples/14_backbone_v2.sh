#!/bin/bash
# Example 14: Multi-Head MobileNetV2 backbone
# Uses a V2-only alpha (0.35) to prove per-version alpha validation:
# the same alpha would be rejected under --backbone v3 because V3
# does not support 0.35.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 14: Multi-Head MobileNetV2 (--backbone v2, alpha=0.35)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/backbone_v2
OUT_NAME=mnv2_multi

python ../src/create_quantized_mobilenet.py \
    --backbone v2 \
    --alpha 0.35 \
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
echo "Example 14 complete."
