#!/bin/bash
# Example 17: Add a head to a trained model with the backbone frozen.
#
# Builds a 3-head model, then dynamically attaches a 4th head that taps
# a different backbone stride (16 instead of the default 32). The
# backbone is frozen, so retraining would only touch the new head.

set -e
cd "$(dirname "$0")"

echo "========================================="
echo "Example 17: Dynamic head addition (frozen backbone)"
echo "========================================="
echo ""

OUT_DIR=../output/examples/dynamic_head
mkdir -p "$OUT_DIR"

python <<'PY'
import sys
sys.path.insert(0, '..')

from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import HeadConfiguration
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

# Build an initial 3-head model on V3, all at the default tap (stride 32).
heads = [
    HeadConfiguration(name=f"head_{i+1}", num_classes=c, activation="linear")
    for i, c in enumerate([5, 2, 3])
]
arch = MultiHeadMobileNetV3QATArchitecture(MultiHeadModelConfig(
    input_shape=(96, 96, 3),
    head_configs=heads,
    arch_params={'alpha': 0.25, 'use_pretrained': False},
    separable_weights=True,
))
arch.get_model()
print(f"Initial heads ({len(arch.head_names)}): {arch.head_names}")

# Dynamically add a 4th head at a different stride with backbone frozen.
new_model = arch.add_head_dynamically(
    num_classes=7,
    head_name="head_4_at_stride_16",
    tap_stride=16,
    freeze_backbone=True,
)
print(f"After add ({len(arch.head_names)}): {arch.head_names}")
print(f"Backbone trainable: {arch.backbone.trainable} (False = only new head trains)")

print("\nFinal model outputs:")
for name, out in new_model.output.items():
    print(f"  {name}: shape={out.shape}")
PY

echo ""
echo "Example 17 complete."
