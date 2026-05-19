# Architecture

How multi-head MobileNet models work and what makes them efficient.

## Overview

The multi-head MobileNet pipeline uses a shared backbone (feature extractor) with multiple task heads. The backbone processes the input image once, and each head makes predictions based on the same features.

## Backbone: MobileNet V1 / V2 / V3-Small / V4

The backbone is one of four MobileNet variants, selected via `--backbone {v1,v2,v3,v4}` (default `v3`). Each is optimized for mobile and edge devices, and all share a common pattern: depthwise-separable or universal-inverted-bottleneck blocks, mobile-friendly activations, and a width multiplier (`alpha`) that controls model capacity.

| Backbone | Source | Alphas | ImageNet pretrained | Notes |
|---|---|---|---|---|
| `v1` | `tf.keras.applications.MobileNet` | 0.25, 0.5, 0.75, 1.0 | Yes (RGB) | Depthwise-separable convs only. |
| `v2` | `tf.keras.applications.MobileNetV2` | 0.35, 0.5, 0.75, 1.0, 1.3, 1.4 | Yes (RGB) | Inverted residuals + linear bottlenecks. |
| `v3` | `tf.keras.applications.MobileNetV3Small` | 0.25, 0.5, 0.75, 1.0 | Yes (RGB, alpha 0.75/1.0 only) | SE blocks + hard-swish. **Default.** |
| `v4` | Custom UIB-based `MobileNetV4ConvS` | 0.25, 0.5, 0.75, 1.0 (interpreted as width multiplier) | No | Universal Inverted Bottleneck blocks. |

The backbone extracts features from the input image and produces a feature map. This happens once, regardless of how many heads you have.

## Classification Heads

Each head is a small neural network attached to the backbone output:

1. Global Average Pooling - reduces spatial dimensions to a single vector
2. Optional Dropout - prevents overfitting
3. Dense layer - final classification with linear activation (outputs logits; softmax applied in post-processing for Vela compatibility)

All heads share the same backbone features but have separate final layers. This means:
- Adding more heads adds minimal computation
- All predictions use the same underlying features
- Model size grows linearly with number of heads

**Note**: Models default to linear activation (no softmax) for Vela compiler compatibility. Apply softmax in post-processing when interpreting outputs.

## Model Size

Model size depends on:

**Backbone**: Determined by `--backbone` and `--alpha` together. Approximate parameter counts (backbone only):

| alpha | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| 0.25 | ~250K | — | ~100K | ~250K |
| 0.35 | — | ~250K | — | — |
| 0.50 | ~1.0M | ~500K | ~400K | ~1.0M |
| 0.75 | ~2.25M | ~1.1M | ~900K | ~2.25M |
| 1.0 | ~4.0M | ~2.2M | ~1.5M | ~4.0M |
| 1.3 | — | ~3.8M | — | — |
| 1.4 | — | ~4.4M | — | — |

V3-Small is the smallest at every alpha; V1 and V4 are roughly comparable; V2 is in between.

**Heads**: Each head adds roughly ~1000 parameters per class
- Head with 2 classes: ~2K parameters
- Head with 10 classes: ~10K parameters
- Head with 100 classes: ~100K parameters

**Quantization**: Reduces size by ~4x
- FP32 weights: 4 bytes per parameter
- uint8/int8 weights: 1 byte per parameter
- Plus quantization overhead (scales, zero points)

Total quantized size ≈ (backbone_params + head_params) / 4 + overhead

## Input Processing

The model expects:
- Input shape: (batch, height, width, channels)
- Channels: 1 (grayscale) or 3 (RGB)
- Pixel values: Typically normalized to [0, 1] or [0, 255]

After quantization, inputs must be uint8. Preprocessing should convert your input images to match the expected format.

## Output Structure

For multi-head models, outputs are a dictionary:
```python
{
    'head_1': <tensor with shape (batch, num_classes_1)>,
    'head_2': <tensor with shape (batch, num_classes_2)>,
    ...
    'unified_heads': <tensor with shape (batch, total_classes)>  # Optional
}
```

Each output tensor contains logits (linear activation, no softmax). After quantization, outputs are uint8 but represent quantized float logits. Use quantization parameters to convert back to float logits, then apply softmax to get probabilities. Models are Vela-compatible by default.

### Unified Output (Optional)

When the `--unified-output` flag is enabled, models include an additional `unified_heads` output that concatenates all head outputs in order. This is useful for:

- **NPU compatibility**: Some NPUs don't support multiple output tensors
- **Simplified inference**: Single tensor access for all predictions
- **Training flexibility**: Individual heads remain accessible for loss computation

**Example**: For a model with heads [5, 2, 3] classes:
- Individual outputs: `head_1` (5 classes), `head_2` (2 classes), `head_3` (3 classes)
- Unified output: `unified_heads` (10 classes: [5, 2, 3] concatenated)

The head order in unified output matches the order specified during model creation and is documented in the model report.

## Performance Characteristics

**Inference Speed**:
- Backbone processing: ~90% of total time
- Head processing: ~10% of total time (shared across all heads)
- Adding heads has minimal impact on speed

**Memory Usage**:
- Backbone features: Main memory consumer
- Head weights: Small compared to backbone
- Activation memory: Depends on input size

**Accuracy Trade-offs**:
- Smaller alpha: Faster, smaller, less accurate
- Larger alpha: Slower, larger, more accurate
- Pretrained weights: Better accuracy, larger model, requires RGB input

## Comparison with Separate Models

Using one multi-head model instead of multiple single-head models:

**Advantages**:
- Shared feature extraction (compute once)
- Smaller total model size
- Consistent features across tasks
- Single inference pass

**Disadvantages**:
- Tasks must be compatible (similar image types)
- All heads use same feature resolution
- Less flexibility in architecture per task

Multi-head models work well when tasks are related (e.g., person detection + gender classification). For unrelated tasks, separate models might be better.
