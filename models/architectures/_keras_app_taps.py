"""
Shared helper for exposing intermediate feature maps from a Keras-applications
backbone (MobileNet / MobileNetV2 / MobileNetV3Small) as stride-keyed tensors
on the parent multi-head model.

Used by the multi-head V1 / V2 / V3 architecture classes via
``_features_by_stride()``. Discovers stride taps by walking the backbone's
internal Keras layers and recording, for each desired stride S, the *last*
layer whose spatial output is ``input_size / S`` — i.e. the deepest layer
at that resolution, immediately before the next downsample.

V4 uses its own (custom UIB) build path and does not go through this helper.
"""

from typing import Dict, Tuple
import tensorflow as tf


def features_by_stride_from_keras_model(
    backbone: tf.keras.Model,
    input_tensor: tf.Tensor,
    input_shape: Tuple[int, int, int],
    desired_strides: Tuple[int, ...] = (4, 8, 16, 32),
) -> Dict[int, tf.Tensor]:
    """Walk the backbone and return a {stride: tensor} dict for desired strides.

    The returned tensors are computed in the *parent* graph (i.e. they flow
    from ``input_tensor``), so they can be fed directly into per-head builders.

    Strides not actually present in the backbone are silently omitted from the
    result; the caller validates against the head's requested ``tap_stride``.
    """
    input_h = input_shape[0]
    if input_h is None:
        return {}

    # Walk layers; for each layer whose output is 4D and divides input_h
    # cleanly, record its name under the corresponding stride. Later layers
    # at the same stride overwrite earlier ones (we want the deepest tap
    # before a downsample).
    stride_to_layer_name: Dict[int, str] = {}
    for layer in backbone.layers:
        # Modern Keras layers don't expose .output_shape; use the symbolic
        # output tensor's shape instead. Layers with multiple outputs return
        # a list of tensors — skip those.
        try:
            output_attr = layer.output
        except (AttributeError, RuntimeError):
            continue
        if isinstance(output_attr, (list, tuple)):
            continue
        try:
            shape = tuple(output_attr.shape)
        except (AttributeError, ValueError):
            continue
        if len(shape) != 4:
            continue
        h = shape[1]
        if h is None or h <= 0:
            continue
        if input_h % h != 0:
            continue
        stride = input_h // h
        if stride in desired_strides:
            stride_to_layer_name[stride] = layer.name

    if not stride_to_layer_name:
        return {}

    # Build a transient multi-output Keras model over the backbone, then
    # apply it to the parent input_tensor so the returned tensors are in the
    # parent graph.
    sorted_items = sorted(stride_to_layer_name.items())
    outputs = [backbone.get_layer(name).output for _, name in sorted_items]
    multi_out_model = tf.keras.Model(inputs=backbone.input, outputs=outputs)
    feats = multi_out_model(input_tensor)
    if not isinstance(feats, list):
        feats = [feats]
    return {stride: tensor for (stride, _), tensor in zip(sorted_items, feats)}
