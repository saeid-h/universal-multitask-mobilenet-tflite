#!/bin/bash
# Example 20: FPN fusion (cross-scale feature pyramid).
#
# Single-tap feature taps (examples 16, 17) route each head to its own
# backbone stride directly. --fusion=fpn instead builds a top-down FPN
# over the strides actually used by heads: each used level gets a
# lateral 1x1 conv, and a top-down upsample-and-add pathway enriches
# finer levels with coarser semantic content before heads see them.
#
# This adds parameters but produces feature maps that are "semantically
# strong AND spatially precise" at every level -- useful for fine
# segmentation and small-object detection.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 20: FPN fusion (--fusion=fpn)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/fpn_fusion
OUT_NAME=fpn_fusion

python ../src/create_quantized_mobilenet.py \
    --backbone v3 \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "5@32,2@16,3@8" \
    --head-names "object_class,person_detection,age_group" \
    --fusion fpn \
    --fpn-channels 64 \
    --output-dir "$OUT_DIR" \
    --output-name "$OUT_NAME" \
    --no-save-keras

echo ""
echo "Verifying TFLite output structure..."
python _verify_tflite.py "$OUT_DIR/${OUT_NAME}_int8.tflite" --expected-heads 3

echo ""
echo "Example 20 complete."
