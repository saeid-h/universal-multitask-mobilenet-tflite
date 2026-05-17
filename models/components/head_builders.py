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