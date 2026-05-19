# Pretrained Weights Reference

This document summarizes pretrained-weight availability across the four supported MobileNet backbones (V1, V2, V3-Small, V4). The detailed tables below focus on the default V3-Small backbone; for the at-a-glance per-backbone summary see [API Reference → `--use-pretrained`](api_reference.md#--use-pretrained).

**Per-backbone summary**:

| Backbone | ImageNet pretrained | Constraints |
|---|---|---|
| `v1` | Yes | All alphas (0.25, 0.5, 0.75, 1.0), RGB only |
| `v2` | Yes | All alphas (0.35, 0.5, 0.75, 1.0, 1.3, 1.4), RGB only |
| `v3` | Yes | Alpha **0.75 or 1.0 only**, RGB only (Keras limitation) |
| `v4` | **No** | Custom UIB-based backbone; no pretrained weights bundled |

All backbones disable pretrained weights for grayscale (1-channel) inputs.

## Keras vs TFLite for QAT Models

| Weight Type | Use for QAT? | Reason |
|-------------|--------------|--------|
| **Keras** (FP32) | ✅ Yes | Standard approach: load pretrained weights to initialize the model, then run quantization-aware training. |
| **TFLite** (int8/uint8) | ❌ No | Already quantized, inference-only format. Cannot be loaded into a Keras model for training. |

## Pretrained Weight Sources

### TensorFlow/Keras (`tf.keras.applications`)

The tool uses `tf.keras.applications.MobileNetV3Small` for the backbone. ImageNet pretrained weights are available for:

| Alpha | Pretrained | Params | Top-1 Acc |
|-------|------------|--------|-----------|
| 0.75 | ✅ | ~2.4M | 65.4% |
| 1.0 | ✅ | ~2.9M | 68.1% |
| 0.25 | ❌ | — | — |
| 0.50 | ❌ | — | — |

**Constraint:** With `weights='imagenet'`, only alpha 0.75 and 1.0 are supported (Keras limitation).

### Channels Requirement

| Channels | Pretrained? | Reason |
|----------|-------------|--------|
| **3 (RGB)** | ✅ Yes | ImageNet weights expect 3 input channels. |
| **1 (grayscale)** | ❌ No | First conv expects 3 channels; shape mismatch. |

**Workaround for grayscale:** Stack the grayscale image 3× to fake RGB, or train from scratch.

### Input Size

With `include_top=False`, backbone weights are spatial-size agnostic. All supported input sizes (96, 128, 160, 224, 256) work with pretrained weights for a given alpha.

## Configurations with Pretrained Weights

Only the following MobileNetV3 configurations can be initialized with pretrained ImageNet weights:

| Input size | Alpha | Channels | Feature vector size | Weight source | Weight size (FP32) | QAT model size (TFLite) |
|------------|-------|----------|---------------------|---------------|--------------------|-------------------------|
| 96×96 | 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 96×96 | 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |
| 128×128 | 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 128×128 | 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |
| 160×160 | 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 160×160 | 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |
| 224×224 | 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 224×224 | 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |
| 256×256 | 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 256×256 | 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |

### Compact Summary

| Alpha | Channels | Feature vector size | Weight source | Weight size (FP32) | QAT model size (TFLite) |
|-------|----------|---------------------|---------------|--------------------|-------------------------|
| 0.75 | 3 | 432 | Keras ImageNet | ~3.5 MB | ~235–255 KB |
| 1.0 | 3 | 576 | Keras ImageNet | ~6 MB | ~385–405 KB |

### Feature Vector Size

The **feature vector** is the backbone output after encoding and before the classification head—i.e., the 1D vector produced by Global Average Pooling on the backbone's last feature map. It is independent of input resolution and depends only on alpha:

- **Alpha 0.75:** 432 dimensions
- **Alpha 1.0:** 576 dimensions

## Configurations Without Pretrained Weights

| Config | Reason |
|--------|--------|
| Grayscale (1 channel) | ImageNet weights expect 3 channels |
| Alpha 0.25 or 0.50 | No ImageNet checkpoints in Keras |
| `--use-pretrained` with alpha 0.25/0.50 | Script enforces alpha 0.75 or 1.0 |

## Alpha 0.5 Pretrained (External)

`mobilenet_v3_small_050_imagenet` exists on [Hugging Face](https://huggingface.co/keras/mobilenet_v3_small_050_imagenet) (278.78K params). To use it would require:

1. Adding `keras_cv` or `keras_hub` as a dependency
2. Loading that backbone instead of `tf.keras.applications.MobileNetV3Small`
3. Mapping weights into the multi-head backbone

The current CLI does not support this.

## Usage Examples

```bash
# Pretrained works
python src/create_quantized_mobilenet.py \
  --alpha 0.75 --input-shape "224x224x3" --heads "5,2" \
  --output-dir ./models --use-pretrained

# Pretrained works
python src/create_quantized_mobilenet.py \
  --alpha 1.0 --input-shape "128x128x3" --heads "2" \
  --output-dir ./models --use-pretrained

# Pretrained does NOT work (alpha 0.25)
python src/create_quantized_mobilenet.py \
  --alpha 0.25 --input-shape "224x224x3" --heads "2" \
  --output-dir ./models --use-pretrained  # Error

# Pretrained does NOT work (grayscale)
python src/create_quantized_mobilenet.py \
  --alpha 0.75 --input-shape "224x224x1" --heads "2" \
  --output-dir ./models --use-pretrained  # Error
```

## Size Notes

- **QAT model sizes** assume a small head (e.g., 2–10 classes). Add ~0.5–2 KB per head for larger classes.
- **Weight size (FP32)** is the backbone pretrained checkpoint size before quantization.
