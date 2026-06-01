#!/bin/bash
# Example 18: int8 calibration with real images via --calibration-data-dir.
#
# By default the tool feeds random data to the TFLite int8 calibrator. For
# production deployments, the calibration data should be representative of
# real inference inputs to get good quantization scales. This example
# demonstrates using a directory of real images.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 18: Real-image calibration (--calibration-data-dir)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/real_calibration
OUT_NAME=real_calib
CALIB_DIR=../examples/demo_data

# Use the bundled demo_data dir as a stand-in for real calibration images.
# In a production run this would point at a held-out validation slice of
# your actual training data, sized in the thousands.
if [ ! -d "$CALIB_DIR" ] || [ -z "$(find "$CALIB_DIR" -name '*.png' -o -name '*.jpg' 2>/dev/null | head -1)" ]; then
    echo "No images found in $CALIB_DIR; generating 16 synthetic PNGs for the demo."
    mkdir -p "$CALIB_DIR/calibration_synth"
    python <<'PY'
import numpy as np
from PIL import Image
np.random.seed(42)
import os
out = "../examples/demo_data/calibration_synth"
for i in range(16):
    arr = (np.random.rand(96, 96, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(os.path.join(out, f"calib_{i:02d}.png"))
PY
    CALIB_DIR=../examples/demo_data/calibration_synth
fi

echo "Calibration images from: $CALIB_DIR"
echo ""

python ../src/create_quantized_mobilenet.py \
    --backbone v3 \
    --alpha 0.25 \
    --input-shape "96x96x3" \
    --heads "5,2,3" \
    --output-dir "$OUT_DIR" \
    --output-name "$OUT_NAME" \
    --no-save-keras \
    --calibration-samples 50 \
    --calibration-data-dir "$CALIB_DIR"

echo ""
echo "Verifying TFLite output structure..."
python _verify_tflite.py "$OUT_DIR/${OUT_NAME}_int8.tflite" --expected-heads 3

echo ""
echo "Example 18 complete."
