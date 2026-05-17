"""
Head builder registry for multi-head architectures.

This module provides a registry system for different head types, allowing
extensible multi-head models with various task-specific head architectures.
"""

from typing import Callable, Dict, Any
import tensorflow as tf
from tensorflow.keras import layers

from .head_configuration import HeadConfiguration

# Type definition for head builder functions
HeadBuilderFn = Callable[[HeadConfiguration, tf.Tensor], tf.Tensor]

# Global registry of head builders
_HEAD_BUILDERS: Dict[str, HeadBuilderFn] = {}


def register_head(head_type: str):
    """Decorator to register a head builder function.
    
    Args:
        head_type: String identifier for the head type
        
    Returns:
        Decorator function
        
    Example:
        @register_head("embedding")
        def build_embedding_head(config, backbone_output):
            # ... implementation
            return output_tensor
    """
    def decorator(fn: HeadBuilderFn):
        if head_type in _HEAD_BUILDERS:
            raise ValueError(f"Head type '{head_type}' is already registered")
        _HEAD_BUILDERS[head_type] = fn
        return fn
    return decorator


def build_head_for_type(head_type: str, head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build a head using the registered builder for the given type.
    
    Args:
        head_type: Type of head to build
        head_config: Configuration for the head
        backbone_output: Output tensor from the backbone
        
    Returns:
        Output tensor from the head
        
    Raises:
        ValueError: If head_type is not registered
    """
    if head_type not in _HEAD_BUILDERS:
        available_types = list(_HEAD_BUILDERS.keys())
        raise ValueError(f"Unknown head_type '{head_type}'. Available types: {available_types}")
    
    return _HEAD_BUILDERS[head_type](head_config, backbone_output)


def get_registered_head_types() -> list:
    """Get list of all registered head types.
    
    Returns:
        List of registered head type names
    """
    return list(_HEAD_BUILDERS.keys())


def get_head_type_info(head_type: str) -> Dict[str, Any]:
    """Get information about a registered head type.
    
    Args:
        head_type: Head type to query
        
    Returns:
        Dictionary with head type information
        
    Raises:
        ValueError: If head_type is not registered
    """
    if head_type not in _HEAD_BUILDERS:
        raise ValueError(f"Unknown head_type '{head_type}'")
    
    builder_fn = _HEAD_BUILDERS[head_type]
    return {
        'head_type': head_type,
        'function_name': builder_fn.__name__,
        'docstring': builder_fn.__doc__,
        'module': builder_fn.__module__
    }


# =============================================================================
# Built-in Head Builders
# =============================================================================

@register_head("standard")
def build_standard_classification(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build standard classification head (GAP + Dropout + Dense).
    
    This is the default head type, equivalent to the original implementation.
    Suitable for single-label multi-class classification tasks.
    
    Args:
        head_config: Head configuration
        backbone_output: Backbone feature map tensor
        
    Returns:
        Classification logits tensor
    """
    x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    return layers.Dense(
        head_config.num_classes,
        activation=head_config.activation,
        name=f"{head_config.name}_output"
    )(x)


@register_head("multilabel")
def build_multilabel_classification(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build multi-label classification head.
    
    Uses sigmoid activation for independent binary classification of each label.
    Suitable for tasks where multiple labels can be active simultaneously
    (e.g., image tags: "indoor", "person", "dog").
    
    Args:
        head_config: Head configuration (activation is overridden to 'sigmoid')
        backbone_output: Backbone feature map tensor
        
    Returns:
        Multi-label probabilities tensor (each value in [0,1])
    """
    x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Always use sigmoid for multi-label (override config activation)
    return layers.Dense(
        head_config.num_classes,
        activation="sigmoid",
        name=f"{head_config.name}_output"
    )(x)


@register_head("regression")
def build_regression_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build regression head for continuous value prediction.
    
    Supports scalar or multi-output regression with optional output activation
    for bounded regression (e.g., values in [0,1] or [-1,1]).
    
    Custom parameters:
        - n_outputs (int): Number of regression outputs (default: head_config.num_classes)
        - output_activation (str): Final activation ('sigmoid', 'tanh', None)
        - output_scale (float): Scale factor for outputs
        - output_bias (float): Bias added to outputs
    
    Args:
        head_config: Head configuration
        backbone_output: Backbone feature map tensor
        
    Returns:
        Regression output tensor
    """
    # Get custom parameters
    n_outputs = head_config.custom_params.get("n_outputs", head_config.num_classes)
    output_activation = head_config.custom_params.get("output_activation", None)
    output_scale = head_config.custom_params.get("output_scale", 1.0)
    output_bias = head_config.custom_params.get("output_bias", 0.0)
    
    x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Linear dense layer
    x = layers.Dense(n_outputs, activation="linear", name=f"{head_config.name}_dense")(x)
    
    # Apply output activation if specified
    if output_activation:
        x = layers.Activation(output_activation, name=f"{head_config.name}_activation")(x)
    
    # Apply scaling and bias if specified
    if output_scale != 1.0 or output_bias != 0.0:
        x = layers.Lambda(
            lambda t: t * output_scale + output_bias,
            name=f"{head_config.name}_scale_bias"
        )(x)
    
    return x


@register_head("embedding")
def build_embedding_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build embedding head for metric learning and similarity matching.
    
    Produces L2-normalized embeddings suitable for cosine similarity computation.
    Commonly used for face recognition, image retrieval, and re-identification tasks.
    
    Custom parameters:
        - embed_dim (int): Embedding dimension (default: 128)
        - use_bn (bool): Use batch normalization before L2 normalization (default: False)
        - projection_layers (list): Hidden layer sizes for MLP projection (default: none)
    
    Args:
        head_config: Head configuration
        backbone_output: Backbone feature map tensor
        
    Returns:
        L2-normalized embedding tensor
    """
    # Get custom parameters
    embed_dim = head_config.custom_params.get("embed_dim", 128)
    use_bn = head_config.custom_params.get("use_bn", False)
    projection_layers = head_config.custom_params.get("projection_layers", [])
    
    x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Optional projection MLP
    for i, hidden_dim in enumerate(projection_layers):
        x = layers.Dense(hidden_dim, activation="relu", name=f"{head_config.name}_proj_{i}")(x)
        if head_config.dropout_rate > 0:
            x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_proj_drop_{i}")(x)
    
    # Final embedding layer
    x = layers.Dense(embed_dim, activation="linear", name=f"{head_config.name}_embedding")(x)
    
    # Optional batch normalization
    if use_bn:
        x = layers.BatchNormalization(name=f"{head_config.name}_bn")(x)
    
    # L2 normalization for cosine similarity
    return layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=-1),
        name=f"{head_config.name}_l2_norm"
    )(x)


@register_head("ssd_detection")
def build_ssd_detection_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build SSD (Single Shot MultiBox Detector) head for object detection.
    
    Suitable for face detection, object detection, and general bounding box regression.
    Predicts both classification scores and bounding box coordinates for multiple
    anchor boxes per spatial location.
    
    Custom parameters:
        - num_anchors (int): Number of anchor boxes per location (default: 3)
        - anchor_scales (list): Scale factors for anchors (default: [0.1, 0.2, 0.37])
        - anchor_ratios (list): Aspect ratios for anchors (default: [0.5, 1.0, 2.0])
        - box_loss_weight (float): Weight for bbox regression loss (default: 1.0)
        - conf_threshold (float): Confidence threshold for NMS (default: 0.5)
    
    Args:
        head_config: Head configuration (num_classes includes background class)
        backbone_output: Backbone feature map tensor
        
    Returns:
        Dictionary with 'boxes' and 'scores' tensors
        - boxes: [batch, num_anchors_total, 4] (x_center, y_center, width, height)
        - scores: [batch, num_anchors_total, num_classes] (class probabilities)
    """
    # Get custom parameters
    num_anchors = head_config.custom_params.get("num_anchors", 3)
    
    B, H, W, C = backbone_output.shape
    num_locations = H * W
    num_anchors_total = num_locations * num_anchors
    
    # Shared convolutional layers for feature processing
    x = layers.Conv2D(256, 3, padding='same', activation='relu', 
                      name=f"{head_config.name}_conv1")(backbone_output)
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      name=f"{head_config.name}_conv2")(x)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Classification head: predicts class probabilities for each anchor
    cls_conv = layers.Conv2D(
        num_anchors * head_config.num_classes, 
        3, padding='same', activation='linear',
        name=f"{head_config.name}_cls_conv"
    )(x)
    
    # Reshape to [batch, num_anchors_total, num_classes]
    cls_scores = layers.Reshape(
        (num_anchors_total, head_config.num_classes),
        name=f"{head_config.name}_cls_reshape"
    )(cls_conv)
    
    # Apply sigmoid for binary classification (face/no-face) or softmax for multi-class
    if head_config.num_classes == 1:
        # Binary detection (face/no-face)
        cls_scores = layers.Activation('sigmoid', name=f"{head_config.name}_cls_sigmoid")(cls_scores)
    else:
        # Multi-class detection
        cls_scores = layers.Activation('softmax', name=f"{head_config.name}_cls_softmax")(cls_scores)
    
    # Box regression head: predicts bbox offsets for each anchor
    box_conv = layers.Conv2D(
        num_anchors * 4,  # 4 coordinates per box
        3, padding='same', activation='linear',
        name=f"{head_config.name}_box_conv"
    )(x)
    
    # Reshape to [batch, num_anchors_total, 4]
    box_preds = layers.Reshape(
        (num_anchors_total, 4),
        name=f"{head_config.name}_box_reshape"
    )(box_conv)
    
    # Return as dictionary for multi-output
    return {
        'scores': cls_scores,
        'boxes': box_preds
    }


@register_head("yolo_detection") 
def build_yolo_detection_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build YOLO-style detection head.
    
    Alternative to SSD with different anchor-free or anchor-based approach.
    Predicts objectness, class probabilities, and bounding boxes.
    
    Custom parameters:
        - grid_size (int): Output grid size (default: inferred from feature map)
        - num_boxes (int): Number of boxes per grid cell (default: 3)
        - coord_scale (float): Scaling factor for coordinates (default: 1.0)
    
    Args:
        head_config: Head configuration
        backbone_output: Backbone feature map tensor
        
    Returns:
        YOLO prediction tensor [batch, grid_h, grid_w, num_boxes * (5 + num_classes)]
        Where 5 = (x, y, w, h, objectness)
    """
    num_boxes = head_config.custom_params.get("num_boxes", 3)
    
    # Feature processing
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                      name=f"{head_config.name}_conv1")(backbone_output)
    x = layers.Conv2D(128, 1, activation='relu',
                      name=f"{head_config.name}_conv2")(x)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Output: objectness (1) + bbox (4) + class probs (num_classes)
    outputs_per_box = 5 + head_config.num_classes
    total_outputs = num_boxes * outputs_per_box
    
    predictions = layers.Conv2D(
        total_outputs,
        1,  # 1x1 conv for final predictions
        activation='linear',
        name=f"{head_config.name}_output"
    )(x)
    
    return predictions


@register_head("segmentation")
def build_segmentation_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build semantic segmentation head.
    
    Performs pixel-wise classification for semantic segmentation tasks.
    Uses transposed convolutions to upsample to input resolution.
    
    Custom parameters:
        - upsample_factor (int): Factor to upsample feature maps (default: 8)
        - intermediate_channels (list): Channels for upsampling layers (default: [256, 128])
        - use_skip_connections (bool): Use U-Net style skip connections (default: False)
    
    Args:
        head_config: Head configuration (num_classes = number of semantic classes)
        backbone_output: Backbone feature map tensor
        
    Returns:
        Segmentation logits [batch, height, width, num_classes]
    """
    upsample_factor = head_config.custom_params.get("upsample_factor", 8)
    intermediate_channels = head_config.custom_params.get("intermediate_channels", [256, 128])
    
    x = backbone_output
    
    # Progressive upsampling with intermediate feature processing
    for i, channels in enumerate(intermediate_channels):
        # Reduce channels
        x = layers.Conv2D(channels, 3, padding='same', activation='relu',
                         name=f"{head_config.name}_upsample_conv_{i}")(x)
        
        if head_config.dropout_rate > 0:
            x = layers.Dropout(head_config.dropout_rate, 
                              name=f"{head_config.name}_upsample_dropout_{i}")(x)
        
        # Upsample by 2x
        x = layers.Conv2DTranspose(channels, 3, strides=2, padding='same', activation='relu',
                                  name=f"{head_config.name}_upsample_{i}")(x)
    
    # Final upsampling to match input resolution
    remaining_factor = upsample_factor // (2 ** len(intermediate_channels))
    if remaining_factor > 1:
        x = layers.Conv2DTranspose(64, 3, strides=remaining_factor, padding='same', activation='relu',
                                  name=f"{head_config.name}_final_upsample")(x)
    
    # Final classification layer
    segmentation_logits = layers.Conv2D(
        head_config.num_classes,
        1,  # 1x1 conv for pixel-wise classification
        activation='linear',
        name=f"{head_config.name}_output"
    )(x)
    
    return segmentation_logits


@register_head("keypoint_detection")
def build_keypoint_detection_head(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build keypoint detection head for pose estimation.
    
    Predicts heatmaps for keypoint locations (e.g., facial landmarks, body joints).
    Each keypoint gets its own heatmap channel.
    
    Custom parameters:
        - heatmap_sigma (float): Gaussian sigma for ground truth heatmaps (default: 1.0)
        - upsample_factor (int): Upsampling factor for heatmaps (default: 4)
        - intermediate_dim (int): Intermediate feature dimension (default: 256)
    
    Args:
        head_config: Head configuration (num_classes = number of keypoints)
        backbone_output: Backbone feature map tensor
        
    Returns:
        Keypoint heatmaps [batch, height, width, num_keypoints]
    """
    upsample_factor = head_config.custom_params.get("upsample_factor", 4)
    intermediate_dim = head_config.custom_params.get("intermediate_dim", 256)
    
    # Feature processing
    x = layers.Conv2D(intermediate_dim, 3, padding='same', activation='relu',
                      name=f"{head_config.name}_conv1")(backbone_output)
    x = layers.Conv2D(intermediate_dim, 3, padding='same', activation='relu',
                      name=f"{head_config.name}_conv2")(x)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # Upsample to higher resolution for precise keypoint localization
    if upsample_factor > 1:
        x = layers.Conv2DTranspose(128, 3, strides=upsample_factor, padding='same', activation='relu',
                                  name=f"{head_config.name}_upsample")(x)
    
    # Generate heatmaps for each keypoint
    heatmaps = layers.Conv2D(
        head_config.num_classes,  # One heatmap per keypoint
        1,  # 1x1 conv for final prediction
        activation='sigmoid',  # Heatmap values in [0,1]
        name=f"{head_config.name}_output"
    )(x)
    
    return heatmaps


@register_head("ordinal")
def build_ordinal_regression(head_config: HeadConfiguration, backbone_output: tf.Tensor) -> tf.Tensor:
    """Build ordinal regression head using CORAL (Consistent Rank Logits).
    
    Suitable for ordered categorical variables like age groups, ratings, severity levels.
    Uses K-1 binary threshold classifiers for K ordinal classes.
    
    The predicted class is: sum(sigmoid_outputs > 0.5) + 1
    
    Custom parameters:
        - threshold_init (str): Initialization for thresholds ('ascending', 'uniform')
    
    Args:
        head_config: Head configuration (num_classes should be >= 2)
        backbone_output: Backbone feature map tensor
        
    Returns:
        Ordinal threshold probabilities tensor (shape: [batch, num_classes-1])
    """
    if head_config.num_classes < 2:
        raise ValueError(f"Ordinal regression requires num_classes >= 2, got {head_config.num_classes}")
    
    threshold_init = head_config.custom_params.get("threshold_init", "ascending")
    
    x = layers.GlobalAveragePooling2D(name=f"{head_config.name}_global_pool")(backbone_output)
    
    if head_config.dropout_rate > 0:
        x = layers.Dropout(head_config.dropout_rate, name=f"{head_config.name}_dropout")(x)
    
    # CORAL: K-1 threshold classifiers
    n_thresholds = head_config.num_classes - 1
    
    # Initialize bias with ascending values to encourage proper ordering
    if threshold_init == "ascending":
        bias_initializer = tf.keras.initializers.Constant([i - n_thresholds/2 for i in range(n_thresholds)])
    else:
        bias_initializer = "zeros"
    
    return layers.Dense(
        n_thresholds,
        activation="sigmoid",
        bias_initializer=bias_initializer,
        name=f"{head_config.name}_output"
    )(x)