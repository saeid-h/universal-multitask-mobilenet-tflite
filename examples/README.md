# Example Scripts

This directory contains example scripts demonstrating various capabilities of the Multi-Head MobileNet quantization tool. Scripts 01–12 use the default V3 backbone; scripts 13–15 demonstrate the other MobileNet versions via `--backbone v1/v2/v4`.

## Quick Start

Run any example script to see the tool in action:

```bash
bash examples/01_basic_single_head.sh
```

## Available Examples

### 01_basic_single_head.sh
Creates a simple binary classification model (e.g., person/no-person detection).
- **Use case**: Binary classification tasks
- **Model**: Single head, 2 classes
- **Size**: Smallest (alpha 0.25)

### 02_multi_head_named.sh
Demonstrates a model that performs multiple tasks simultaneously.
- **Use case**: Multi-task learning
- **Model**: 3 heads with named outputs
- **Tasks**: Object classification, person detection, age grouping

### 03_grayscale_model.sh
Creates a smaller model using grayscale input for memory-constrained devices.
- **Use case**: Resource-constrained devices
- **Input**: 128x128 grayscale (1 channel)
- **Benefits**: 3x smaller input, faster inference

### 04_different_sizes.sh
Demonstrates the size/accuracy trade-off with different alpha values.
- **Creates**: 3 models (alpha 0.25, 0.50, 0.75)
- **Shows**: Size vs accuracy trade-offs
- **Sizes**: ~100KB, ~400KB, ~900KB quantized

### 05_high_resolution.sh
Creates a model for detailed recognition tasks with higher resolution.
- **Input**: 320x320 RGB (high resolution)
- **Use case**: Fine-grained classification
- **Trade-off**: Larger size, better detail recognition

### 06_pretrained_weights.sh
Demonstrates transfer learning with ImageNet pretrained weights.
- **Requirements**: Alpha 0.75 or 1.0, RGB input
- **Benefits**: Better accuracy with limited data
- **Use case**: Similar tasks to ImageNet classification

### 07_custom_calibration.sh
Demonstrates using more calibration samples for better quantization.
- **Calibration**: 300 samples (default: 100)
- **Benefits**: Better quantization accuracy
- **Use when**: Quantization quality is critical

### 08_quantize_trained_model.sh
Demonstrates loading and quantizing an existing trained Keras model.
- **Workflow**: Train → Save → Quantize
- **Use case**: Quantize models trained on your dataset
- **Requirements**: Trained .keras model file

### 09_comprehensive_demo.sh
Runs multiple examples to showcase project capabilities.
- **Creates**: 5 different model types
- **Shows**: Full range of capabilities
- **Purpose**: Quick overview of all features

### 10–12 (additional V3 examples)
- `10_create_128x128_model.sh` — 128×128 grayscale, [5,2,5,3,3] heads.
- `11_separate_tflite_export.sh` — separable weights, mixed-precision per-component export.
- `12_alternative_heads.sh` — non-classification head types (multilabel, regression, embedding, ordinal) via the Python API.

### 13_backbone_v1.sh
Demonstrates `--backbone v1` (MobileNetV1).
- **Backbone**: `tf.keras.applications.MobileNet`, alpha 0.25
- **Heads**: 3 heads (5, 2, 3 classes)
- **Verifier**: Calls `_verify_tflite.py` to assert 3 output tensors and uint8 input.

### 14_backbone_v2.sh
Demonstrates `--backbone v2` (MobileNetV2) with a V2-only alpha.
- **Backbone**: `tf.keras.applications.MobileNetV2`, alpha **0.35** (rejected by V1/V3/V4)
- **Heads**: 3 heads — proves per-version alpha validation works.

### 15_backbone_v4.sh
Demonstrates `--backbone v4` (MobileNetV4-Conv-S, custom UIB-based).
- **Backbone**: project's custom V4 (no Keras-applications V4 exists)
- **Heads**: 3 heads, no pretrained weights.

### _verify_tflite.py
Internal helper used by the backbone-specific examples. Loads a `.tflite` file and asserts the output-tensor count matches `--expected-heads` and the input dtype is `uint8`. Exits non-zero on mismatch.

### create_experiment_models.sh
Creates all experimental models with standardized naming format.
- **Format**: `{resolution}_{channels}_{num_heads}` (e.g., `96_3_5` = 96x96 resolution, 3 channels, 5 heads)
- **Creates**: 8 model variants covering different resolutions, channels, and head counts
- **Output**: Models saved to `output/mnv3_experiments/` and copied to `output/mnv3_exp/`
- **Use case**: Generate multiple model variants for experimentation and comparison

### create_experiment_unified_heads_models.sh
Creates multi-head experimental models with unified output for NPU compatibility.
- **Format**: `{resolution}_{channels}_{num_heads}` (e.g., `96_3_5` = 96x96 resolution, 3 channels, 5 heads)
- **Creates**: 5 multi-head model variants (single-head models skipped)
- **Feature**: All models include `unified_heads` output (concatenated head outputs)
- **Output**: Models saved to `output/mnv3_unified_experiments/` and copied to `output/mnv3_unified_exp/` with `_unified` suffix
- **Use case**: Generate models for NPUs that don't support multiple output tensors

## Running Examples

### Individual Examples

Run any script directly from the project root:

```bash
bash examples/01_basic_single_head.sh
```

Or from the examples directory:

```bash
cd examples
bash 01_basic_single_head.sh
```

### All Examples

Run the comprehensive demo:

```bash
bash examples/09_comprehensive_demo.sh
```

### Custom Examples

Use the scripts as templates for your own use cases:

```bash
# Edit a script to customize parameters
nano examples/01_basic_single_head.sh
```

## Output Location

All example outputs are saved to:
```
../output/examples/<example_name>/
```

The experimental models scripts save to:
- `create_experiment_models.sh`:
  - `output/mnv3_experiments/<model_name>/` - Full model outputs with reports
  - `output/mnv3_exp/` - Copied TFLite models with simplified names
- `create_experiment_unified_heads_models.sh`:
  - `output/mnv3_unified_experiments/<model_name>/` - Full model outputs with reports
  - `output/mnv3_unified_exp/` - Copied TFLite models with `_unified` suffix

Each output directory contains:
- `*_int8.tflite` - Quantized TensorFlow Lite model
- `*_report.json` - Detailed model statistics
- `*_summary.txt` - Human-readable summary
- `*_quantization_info.json` - Quantization analysis

## Common Use Cases

### Smallest Model for Microcontrollers
```bash
bash examples/03_grayscale_model.sh
```

### Multi-Task Learning
```bash
bash examples/02_multi_head_named.sh
```

### Best Accuracy with Pretrained Weights
```bash
bash examples/06_pretrained_weights.sh
```

### Custom Training + Quantization
```bash
# First train
python examples/example_training.py

# Then quantize (update path in script)
bash examples/08_quantize_trained_model.sh
```

### Generate All Experimental Models
```bash
bash examples/create_experiment_models.sh
```

This creates 8 model variants with standardized naming format for experimentation.

### Generate Models with Unified Output
```bash
bash examples/create_experiment_unified_heads_models.sh
```

This creates 5 multi-head model variants with unified output for NPU compatibility.

## Next Steps

1. **Explore Examples**: Run different scripts to see capabilities
2. **Modify Scripts**: Customize for your specific needs
3. **Train Models**: Use `example_training.py` to train custom models
4. **Read Documentation**: See `docs/` directory for detailed guides

## Notes

- All scripts are executable and can be run from the repository root or examples directory
- Scripts create output directories automatically
- Use `--no-save-keras` to skip saving intermediate Keras files (saves disk space)
- Models are Vela-compatible by default for Arm Ethos-U NPU deployment
- The `create_experiment_models.sh` script creates models with standardized naming format for experimentation

