#!/bin/bash
# Example 19: Custom MobileNetV3-Small backbone (--backbone v3c).
#
# The default --backbone v3 uses tf.keras.applications.MobileNetV3Small,
# which only accepts its specific input shapes and has no grayscale
# ImageNet weights. --backbone v3c uses the project's own custom V3-Small
# implementation, which accepts a wider set of input shapes including
# grayscale variants and is the right choice for MCU-class targets that
# want V3-Small features at 1-channel input.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 19: Custom V3-Small (--backbone v3c, 96x96 grayscale)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/backbone_v3c
OUT_NAME=mnv3c_multi

python ../src/create_quantized_mobilenet.py \
    --backbone v3c \
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
echo "Example 19 complete."
