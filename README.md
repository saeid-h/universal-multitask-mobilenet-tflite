# Multi-Head MobileNet Quantization Tool

Create quantized MobileNet models with custom multi-head outputs. This tool generates fully quantized TensorFlow Lite models (uint8 input/output) from MobileNet **V1, V2, V3-Small, and V4** architectures with your choice of task heads (classification, regression, embedding, detection, segmentation, keypoint, OCR, …). Models are Vela-compatible by default for Arm Ethos-U NPU deployment.

Select the backbone with `--backbone {v1,v2,v3,v4}` (default `v3`, which preserves the tool's original behavior).

## Quick Start

Install dependencies:
```bash
pip install tensorflow numpy
```

Create a model with multiple heads:
```bash
python src/create_quantized_mobilenet.py \
    --heads "5,2,3" \
    --output-dir ./models
```

This creates a model with three heads: 5 classes, 2 classes, and 3 classes, saved as a quantized TFLite file.

For NPUs that require a single output tensor, use the `--unified-output` flag to create a concatenated output while keeping individual heads accessible for training.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy

The script uses the existing models package from this repository.

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed setup instructions.

## Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Getting Started](docs/getting_started.md) - Installation and first steps
- [Tutorial](docs/tutorial.md) - Step-by-step guide with examples
- [Training Guide](docs/training_guide.md) - Using as a module and training with multiple datasets
- [API Reference](docs/api_reference.md) - Complete parameter documentation
- [Examples](docs/examples.md) - Various use cases and configurations
- [Architecture](docs/architecture.md) - How multi-head MobileNet works (shared backbone + heads)
- [Head Types Reference](docs/head_types_reference.md) - Classification, regression, embedding, and more
- [Task-Head Reference](docs/task_head_reference.md) - Complete CV task mapping & implementation roadmap
- [Quantization Guide](docs/quantization_guide.md) - Understanding quantization
- [Pretrained Weights Reference](docs/pretrained_weights_reference.md) - Which configs support pretrained weights
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Basic Usage

```bash
python src/create_quantized_mobilenet.py \
    --backbone v3 \
    --alpha 0.25 \
    --input-shape "224x224x3" \
    --heads "2,5" \
    --output-dir ./output
```

Try a different backbone:

```bash
# MobileNetV1 multi-head model
python src/create_quantized_mobilenet.py --backbone v1 --alpha 0.25 --heads "5,2,3" --output-dir ./output

# MobileNetV2 with a V2-only alpha (0.35)
python src/create_quantized_mobilenet.py --backbone v2 --alpha 0.35 --heads "5,2,3" --output-dir ./output

# MobileNetV4 (custom UIB backbone; no ImageNet pretrained weights)
python src/create_quantized_mobilenet.py --backbone v4 --alpha 0.25 --heads "5,2,3" --output-dir ./output
```

Output filenames are auto-prefixed with the backbone version (`mnv1_*`, `mnv2_*`, `mnv3_*`, `mnv4_*`).

See [docs/getting_started.md](docs/getting_started.md) for more details.
