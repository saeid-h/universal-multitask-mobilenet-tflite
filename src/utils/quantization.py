"""Quantization utilities for converting Keras models to TFLite."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import tensorflow as tf


# Image extensions the directory-based representative dataset will pick up.
_IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def _list_images_in_dir(image_dir: str) -> List[Path]:
    """Return sorted image paths under image_dir (recursive, deduped)."""
    root = Path(image_dir)
    if not root.is_dir():
        raise ValueError(
            f"--calibration-data-dir is not a directory: {image_dir}"
        )
    files = sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )
    if not files:
        raise ValueError(
            f"--calibration-data-dir contains no images "
            f"(supported extensions: {sorted(_IMG_EXTENSIONS)}): {image_dir}"
        )
    return files


def _load_and_preprocess_image(
    path: Path,
    input_shape: Tuple[int, int, int],
) -> np.ndarray:
    """Load an image and shape it into a (1, H, W, C) float32 batch in [0, 1]."""
    h, w, c = input_shape
    raw = tf.io.read_file(str(path))
    # decode_image returns a 3D tensor; force expand_animations=False so GIFs
    # come back as a single frame rather than a 4D animation tensor.
    img = tf.io.decode_image(raw, channels=c, expand_animations=False)
    img = tf.image.resize(img, (h, w), method='bilinear')
    img = tf.cast(img, tf.float32) / 255.0  # match the random-gen [0, 1] range
    return img.numpy().reshape((1, h, w, c)).astype(np.float32)


def create_representative_dataset(
    input_shape: Tuple[int, int, int],
    num_samples: int,
    image_dir: Optional[str] = None,
):
    """
    Create a representative-dataset generator for TFLite int8 calibration.

    Without ``image_dir`` this falls back to the original behavior: random
    float data in [0, 1] matching ``input_shape``. With ``image_dir``,
    images are loaded from that directory, resized to ``input_shape``'s
    spatial dims (RGB or grayscale per channel count), and scaled to
    [0, 1] — yielded one batch at a time.

    Args:
        input_shape: Input shape as (height, width, channels).
        num_samples: Number of calibration samples to yield. If
            ``image_dir`` has fewer images than this, the dataset cycles
            through them (so the calibrator still sees ``num_samples``
            batches).
        image_dir: Optional directory of real calibration images.

    Returns:
        A generator function suitable for
        ``tf.lite.TFLiteConverter.representative_dataset``.
    """
    if image_dir is None:
        def random_generator():
            for _ in range(num_samples):
                data = np.random.random((1,) + input_shape).astype(np.float32)
                yield [data]
        return random_generator

    image_paths = _list_images_in_dir(image_dir)

    def dir_generator():
        for i in range(num_samples):
            path = image_paths[i % len(image_paths)]
            batch = _load_and_preprocess_image(path, input_shape)
            yield [batch]
    return dir_generator


def quantize_to_tflite(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int],
    calibration_samples: int = 100,
    output_path: Optional[str] = None,
    calibration_data_dir: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to quantized TFLite with uint8 input/output.

    Uses a representative dataset for post-training int8 calibration.
    By default the representative data is random in [0, 1]; pass
    ``calibration_data_dir`` to use a directory of real images instead
    (resized + normalized to match the model's expected input).

    Args:
        model: Keras model to quantize.
        input_shape: Input shape as (height, width, channels).
        calibration_samples: Number of calibration batches to feed.
        output_path: Optional path to save the TFLite model.
        calibration_data_dir: Optional directory of real images to use
            as the representative dataset. If fewer images than
            ``calibration_samples`` are present, the dataset cycles.

    Returns:
        Tuple of (TFLite model bytes, quantization analysis dictionary).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    representative_dataset = create_representative_dataset(
        input_shape,
        calibration_samples,
        image_dir=calibration_data_dir,
    )
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save first if path provided, then analyze
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def analyze_tflite_model(
    tflite_model: bytes,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze TFLite model structure and quantization.
    
    Examines the model's tensors, inputs, outputs, and quantization status.
    
    Args:
        tflite_model: TFLite model as bytes
        model_path: Optional path to the model file (for direct loading)
        
    Returns:
        Dictionary containing model analysis information
    """
    if model_path:
        interpreter = tf.lite.Interpreter(model_path=model_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tflite') as f:
            f.write(tflite_model)
            temp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=temp_path)
        finally:
            os.unlink(temp_path)
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    tensor_details = interpreter.get_tensor_details()
    
    int8_count = sum(1 for t in tensor_details if t['dtype'] == np.int8)
    uint8_count = sum(1 for t in tensor_details if t['dtype'] == np.uint8)
    float32_count = sum(1 for t in tensor_details if t['dtype'] == np.float32)
    quantized_count = int8_count + uint8_count
    
    analysis = {
        'input_details': [
            {
                'name': d['name'],
                'shape': d['shape'].tolist(),
                'dtype': str(d['dtype']),
                'quantization': d.get('quantization_parameters', {})
            }
            for d in input_details
        ],
        'output_details': [
            {
                'name': d['name'],
                'shape': d['shape'].tolist(),
                'dtype': str(d['dtype']),
                'quantization': d.get('quantization_parameters', {})
            }
            for d in output_details
        ],
        'quantization': {
            'total_tensors': len(tensor_details),
            'int8_tensors': int8_count,
            'uint8_tensors': uint8_count,
            'float32_tensors': float32_count,
            'fully_quantized': quantized_count > 0 and float32_count == 0
        }
    }
    
    return analysis


def convert_to_int8_tflite(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int],
    calibration_samples: int = 100,
    output_path: Optional[str] = None,
    calibration_data_dir: Optional[str] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to int8 quantized TFLite.

    Args:
        model: Keras model to quantize.
        input_shape: Input shape for calibration.
        calibration_samples: Number of calibration samples.
        output_path: Optional path to save the TFLite model.
        calibration_data_dir: Optional directory of real images used as
            the representative dataset (passed through to
            create_representative_dataset). Only meaningful for the
            backbone or full-model calibration — head calibration
            should stay random because the head consumes feature maps,
            not images.

    Returns:
        Tuple of (TFLite model bytes, analysis dictionary).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    representative_dataset = create_representative_dataset(
        input_shape, calibration_samples, image_dir=calibration_data_dir,
    )
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def convert_to_fp16_tflite(
    model: tf.keras.Model,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to fp16 TFLite.
    
    Args:
        model: Keras model to convert
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def convert_to_fp32_tflite(
    model: tf.keras.Model,
    output_path: Optional[str] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Convert Keras model to fp32 TFLite.
    
    Args:
        model: Keras model to convert
        output_path: Optional path to save the TFLite model
        
    Returns:
        Tuple of (TFLite model bytes, analysis dictionary)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    tflite_model = converter.convert()
    
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        analysis = analyze_tflite_model(tflite_model, output_path)
    else:
        analysis = analyze_tflite_model(tflite_model, None)
    
    return tflite_model, analysis


def export_separate_tflite_models(
    architecture,
    input_shape: Tuple[int, int, int],
    output_dir: str,
    base_name: str,
    backbone_format: str = "int8",
    head_format: str = "fp16",
    calibration_samples: int = 100,
    calibration_data_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Export backbone and heads as separate TFLite files with different precisions.

    Args:
        architecture: Multi-head architecture with separable_weights=True.
        input_shape: Input shape for calibration (backbone only).
        output_dir: Directory to save TFLite files.
        base_name: Base name for output files.
        backbone_format: Format for backbone ("int8", "fp16", "fp32").
        head_format: Format for heads ("int8", "fp16", "fp32").
        calibration_samples: Number of calibration samples for int8 conversion.
        calibration_data_dir: Optional directory of real images. Used only
            for the backbone's int8 calibration (heads consume feature
            maps, not images, so their calibration stays random).

    Returns:
        Dictionary with export results and file paths.

    Raises:
        ValueError: If architecture doesn't have separable_weights enabled.
    """
    if not architecture.is_separable:
        raise ValueError(
            "export_separate_tflite_models requires architecture with separable_weights=True"
        )
    
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'backbone': {},
        'heads': {},
        'files': []
    }
    
    # Convert backbone
    backbone_model = architecture.backbone
    backbone_file = output_path / f"{base_name}_backbone_{backbone_format}.tflite"
    
    if backbone_format == "int8":
        _, backbone_analysis = convert_to_int8_tflite(
            backbone_model, input_shape, calibration_samples, str(backbone_file),
            calibration_data_dir=calibration_data_dir,
        )
    elif backbone_format == "fp16":
        _, backbone_analysis = convert_to_fp16_tflite(
            backbone_model, str(backbone_file)
        )
    elif backbone_format == "fp32":
        _, backbone_analysis = convert_to_fp32_tflite(
            backbone_model, str(backbone_file)
        )
    else:
        raise ValueError(f"Unsupported backbone format: {backbone_format}")
    
    backbone_size = backbone_file.stat().st_size
    results['backbone'] = {
        'file': str(backbone_file),
        'format': backbone_format,
        'size_bytes': backbone_size,
        'size_kb': backbone_size / 1024,
        'analysis': backbone_analysis
    }
    results['files'].append(str(backbone_file))
    
    # Convert heads
    for head_name in architecture.head_names:
        head_model = architecture.head(head_name)
        head_file = output_path / f"{base_name}_{head_name}_{head_format}.tflite"
        
        if head_format == "int8":
            # For heads, we need to get the feature shape from backbone output
            dummy_input = tf.zeros((1,) + input_shape)
            backbone_output = backbone_model(dummy_input)
            feature_shape = backbone_output.shape[1:]
            _, head_analysis = convert_to_int8_tflite(
                head_model, feature_shape, calibration_samples, str(head_file)
            )
        elif head_format == "fp16":
            _, head_analysis = convert_to_fp16_tflite(
                head_model, str(head_file)
            )
        elif head_format == "fp32":
            _, head_analysis = convert_to_fp32_tflite(
                head_model, str(head_file)
            )
        else:
            raise ValueError(f"Unsupported head format: {head_format}")
        
        head_size = head_file.stat().st_size
        results['heads'][head_name] = {
            'file': str(head_file),
            'format': head_format,
            'size_bytes': head_size,
            'size_kb': head_size / 1024,
            'analysis': head_analysis
        }
        results['files'].append(str(head_file))
    
    return results


def validate_model_outputs(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    """
    Validate model structure and test inference.
    
    Runs a test inference to verify the model works correctly and extracts
    output information.
    
    Args:
        model: Keras model to validate
        input_shape: Input shape as (height, width, channels)
        
    Returns:
        Dictionary containing output information (shapes, dtypes)
    """
    test_input = np.random.random((1,) + input_shape).astype(np.float32)
    outputs = model(test_input)
    
    output_info = {}
    if isinstance(outputs, dict):
        for head_name, output in outputs.items():
            output_info[head_name] = {
                'shape': list(output.shape),
                'dtype': str(output.dtype)
            }
    else:
        output_info['output'] = {
            'shape': list(outputs.shape),
            'dtype': str(outputs.dtype)
        }
    
    return output_info

