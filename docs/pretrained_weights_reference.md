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

## Pretrained-Weight Catalog (sorted)

Comprehensive table of every ImageNet-pretrained MobileNet checkpoint Keras can download automatically when you pass `--use-pretrained` (RGB only). All URLs were verified against Google's CDN. Rows are sorted by Backbone → Alpha → Input size. Sizes that are missing for a given alpha are not in the catalog — try a row that is.

| Backbone | Alpha | Input | Backbone params | Feature dim | Top-1 @ 224 | Download URL |
|---|---|---|---|---|---|---|
| v1 | 0.25 | 128 | 218,544 | 256 | 50.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_128_tf_no_top.h5> |
| v1 | 0.25 | 160 | 218,544 | 256 | 50.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_160_tf_no_top.h5> |
| v1 | 0.25 | 192 | 218,544 | 256 | 50.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_192_tf_no_top.h5> |
| v1 | 0.25 | 224 | 218,544 | 256 | 50.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_2_5_224_tf_no_top.h5> |
| v1 | 0.50 | 128 | 829,536 | 512 | 63.7% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_5_0_128_tf_no_top.h5> |
| v1 | 0.50 | 160 | 829,536 | 512 | 63.7% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_5_0_160_tf_no_top.h5> |
| v1 | 0.50 | 192 | 829,536 | 512 | 63.7% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_5_0_192_tf_no_top.h5> |
| v1 | 0.50 | 224 | 829,536 | 512 | 63.7% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_5_0_224_tf_no_top.h5> |
| v1 | 0.75 | 128 | 1,832,976 | 768 | 68.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_7_5_128_tf_no_top.h5> |
| v1 | 0.75 | 160 | 1,832,976 | 768 | 68.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_7_5_160_tf_no_top.h5> |
| v1 | 0.75 | 192 | 1,832,976 | 768 | 68.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_7_5_192_tf_no_top.h5> |
| v1 | 0.75 | 224 | 1,832,976 | 768 | 68.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_7_5_224_tf_no_top.h5> |
| v1 | 1.00 | 128 | 3,228,864 | 1024 | 70.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_128_tf_no_top.h5> |
| v1 | 1.00 | 160 | 3,228,864 | 1024 | 70.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_160_tf_no_top.h5> |
| v1 | 1.00 | 192 | 3,228,864 | 1024 | 70.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_192_tf_no_top.h5> |
| v1 | 1.00 | 224 | 3,228,864 | 1024 | 70.6% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet/mobilenet_1_0_224_tf_no_top.h5> |
| v2 | 0.35 | 96  | 410,208   | 1280 | 60.3% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_96_no_top.h5> |
| v2 | 0.35 | 128 | 410,208   | 1280 | 60.3% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_128_no_top.h5> |
| v2 | 0.35 | 160 | 410,208   | 1280 | 60.3% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_160_no_top.h5> |
| v2 | 0.35 | 192 | 410,208   | 1280 | 60.3% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_192_no_top.h5> |
| v2 | 0.35 | 224 | 410,208   | 1280 | 60.3% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.35_224_no_top.h5> |
| v2 | 0.50 | 96  | 706,224   | 1280 | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_96_no_top.h5> |
| v2 | 0.50 | 128 | 706,224   | 1280 | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_128_no_top.h5> |
| v2 | 0.50 | 160 | 706,224   | 1280 | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_160_no_top.h5> |
| v2 | 0.50 | 192 | 706,224   | 1280 | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_192_no_top.h5> |
| v2 | 0.50 | 224 | 706,224   | 1280 | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224_no_top.h5> |
| v2 | 0.75 | 96  | 1,382,064 | 1280 | 69.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_96_no_top.h5> |
| v2 | 0.75 | 128 | 1,382,064 | 1280 | 69.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_128_no_top.h5> |
| v2 | 0.75 | 160 | 1,382,064 | 1280 | 69.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_160_no_top.h5> |
| v2 | 0.75 | 192 | 1,382,064 | 1280 | 69.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_192_no_top.h5> |
| v2 | 0.75 | 224 | 1,382,064 | 1280 | 69.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.75_224_no_top.h5> |
| v2 | 1.00 | 96  | 2,257,984 | 1280 | 71.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5> |
| v2 | 1.00 | 128 | 2,257,984 | 1280 | 71.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5> |
| v2 | 1.00 | 160 | 2,257,984 | 1280 | 71.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5> |
| v2 | 1.00 | 192 | 2,257,984 | 1280 | 71.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_192_no_top.h5> |
| v2 | 1.00 | 224 | 2,257,984 | 1280 | 71.8% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5> |
| v2 | 1.30 | 224 | 3,766,048 | 1664 | 74.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.3_224_no_top.h5> |
| v2 | 1.40 | 224 | 4,363,712 | 1792 | 75.0% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5> |
| v3 | 0.75 | 224 | 583,160   | 432  | 65.4% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_0.75_float_no_top_v2.h5> |
| v3 | 1.00 | 224 | 939,120   | 576  | 68.1% | <https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5> |

**Notes**:
- *Backbone params* is the count for `include_top=False` (the model this tool actually uses).
- *Feature dim* is the channel count of the backbone's final feature map.
- *Top-1 @ 224* is the published ImageNet validation accuracy at 224×224 input — figures from the MobileNet V1 (Howard et al. 2017), V2 (Sandler et al. 2018), and V3 (Howard et al. 2019) papers and the Keras MobileNet V3 docstring. Lower input resolutions yield somewhat lower accuracy for the same alpha; use 224 weights when accuracy matters and a smaller input only when you must.
- *Channels*: every checkpoint is RGB (3 channels). Grayscale inputs cannot use these weights; the tool silently disables `--use-pretrained` in that case.
- V2 alphas 1.3 and 1.4 only have a 224 checkpoint; smaller input sizes are not pretrained.
- V3 alphas 0.25 and 0.50 have no Keras checkpoint — see the [Alpha 0.5 Pretrained (External)](#alpha-05-pretrained-external) section below for a third-party option.
- V4 (custom UIB backbone) has no pretrained weights bundled — `--use-pretrained` is silently ignored.

**Auto-fetch behavior**: when `--use-pretrained` is set and `(backbone, alpha, input_size)` matches a catalog row, Keras downloads the URL above into `~/.keras/models/` and loads it. No manual download required. The URL is provided for reference, mirror setups, and offline preparation.

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
