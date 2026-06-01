#!/bin/bash
# Example 16: Per-head feature taps via the --heads "N@stride" syntax.
# Demonstrates routing different heads to different backbone strides.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 16: Per-head feature taps"
echo "========================================="
echo "Each head can independently tap a different backbone stride."
echo ""

OUT_DIR=../output/examples/feature_taps
OUT_NAME=feature_taps

python ../src/create_quantized_mobilenet.py \
    --backbone v3 \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "5@32,2@16,3@8" \
    --head-names "object_class,person_detection,age_group" \
    --output-dir "$OUT_DIR" \
    --output-name "$OUT_NAME" \
    --no-save-keras

echo ""
echo "Verifying TFLite output structure..."
python _verify_tflite.py "$OUT_DIR/${OUT_NAME}_int8.tflite" --expected-heads 3

echo ""
echo "Example 16 complete."
