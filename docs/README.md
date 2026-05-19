# Documentation

This directory contains complete documentation for the Multi-Head MobileNet quantization tool (supports MobileNet V1/V2/V3-Small/V4 backbones via `--backbone`).

## Documentation Structure

- **[Getting Started](getting_started.md)** - Set up and create your first model
- **[Tutorial](tutorial.md)** - Walkthrough with step-by-step examples
- **[Training Guide](training_guide.md)** - Using the model as a module and training with multiple datasets
- **[API Reference](api_reference.md)** - Complete command-line parameter documentation
- **[Examples](examples.md)** - Various use cases and model configurations
- **[Architecture](architecture.md)** - How the multi-head MobileNet pipeline works internally (shared backbone + heads)
- **[Quantization Guide](quantization_guide.md)** - Understanding uint8 quantization
- **[Pretrained Weights Reference](pretrained_weights_reference.md)** - Which configs support pretrained weights, sizes, and feature vectors
- **[Troubleshooting](troubleshooting.md)** - Solutions to common problems

## Quick Navigation

**New to the tool?** Start with [Getting Started](getting_started.md), then follow the [Tutorial](tutorial.md).

**Want quick examples?** Check the example scripts in the `examples/` directory. Each script demonstrates a specific capability.

**Need to train the model?** See the [Training Guide](training_guide.md) for programmatic usage and multi-dataset training.

**Need specific information?** Check the [API Reference](api_reference.md) for parameter details.

**Looking for ideas?** Browse the [Examples](examples.md) for different use cases.

**Running into issues?** See [Troubleshooting](troubleshooting.md) for solutions.

## Quick Start

The main script is located at `src/create_quantized_mobilenet.py`. Basic usage:

```bash
python src/create_quantized_mobilenet.py --backbone v3 --heads "5,2,3" --output-dir ./models
```

Use `--backbone v1`, `v2`, or `v4` to build the same heads on a different MobileNet backbone.

Or try the example scripts:
```bash
bash examples/01_basic_single_head.sh
```
