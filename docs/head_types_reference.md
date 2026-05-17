# Head Types Reference

This document describes all available head types in the multi-head MobileNet architecture and how to use them for different tasks.

## Overview

The multi-head architecture supports extensible head types through a registry system. Each head type is optimized for specific task categories and automatically handles appropriate loss functions and metrics.

## Available Head Types

### 1. Standard Classification (`standard`)

**Purpose**: Traditional multi-class classification where each sample belongs to exactly one class.

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_classes, activation)

**Use Cases**:
- Object classification (cat, dog, car, etc.)
- Scene recognition (indoor, outdoor, urban, etc.)
- Quality assessment (good, fair, poor)

**Configuration**:
```python
from models.components.head_configuration import create_classification_head

head = create_classification_head(
    name="scene_type",
    num_classes=5,
    activation="linear",  # or "softmax"
    dropout_rate=0.2
)
```

**Training**:
- **Loss**: `SparseCategoricalCrossentropy(from_logits=True)`
- **Metrics**: `accuracy`, `sparse_top_k_categorical_accuracy`
- **Label format**: Integer class indices (0, 1, 2, ...)

### 2. Multi-Label Classification (`multilabel`)

**Purpose**: Multiple independent binary classifications where samples can have multiple active labels simultaneously.

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_labels, activation='sigmoid')

**Use Cases**:
- Image tagging (indoor + person + furniture)
- Medical diagnosis (multiple conditions present)
- Content moderation (multiple policy violations)

**Configuration**:
```python
from models.components.head_configuration import create_multilabel_head

head = create_multilabel_head(
    name="image_tags",
    num_labels=8,
    dropout_rate=0.3
)
```

**Training**:
- **Loss**: `BinaryCrossentropy(from_logits=False)`
- **Metrics**: `binary_accuracy`, `precision`, `recall`
- **Label format**: Binary array (e.g., [1, 0, 1, 0, 1] for 5 labels)

### 3. Regression (`regression`)

**Purpose**: Predicting continuous numerical values (single or multiple outputs).

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(n_outputs, activation='linear') → Optional scaling/bias

**Use Cases**:
- Age estimation (years)
- Pose estimation (yaw, pitch, roll angles)
- Quality scores (0.0 to 1.0)
- Bounding box coordinates

**Configuration**:
```python
from models.components.head_configuration import create_regression_head

# Single output regression
age_head = create_regression_head(
    name="age_years",
    n_outputs=1,
    output_scale=100.0,  # Scale to 0-100 years
    dropout_rate=0.2
)

# Multi-output regression with bounded activation
pose_head = create_regression_head(
    name="head_pose",
    n_outputs=3,  # yaw, pitch, roll
    output_activation="tanh",  # Bound to [-1, 1]
    dropout_rate=0.1
)

# Quality score (0-1 range)
quality_head = create_regression_head(
    name="blur_score",
    n_outputs=1,
    output_activation="sigmoid",  # Bound to [0, 1]
)
```

**Training**:
- **Loss**: `MeanSquaredError`, `MeanAbsoluteError`, or `Huber`
- **Metrics**: `mae`, `mse`
- **Label format**: Float arrays matching n_outputs

### 4. Embedding (`embedding`)

**Purpose**: Learning feature representations for similarity matching, clustering, or retrieval.

**Architecture**: GlobalAveragePooling2D → Dropout → Optional MLP → Dense(embed_dim) → BatchNorm (optional) → L2Normalize

**Use Cases**:
- Face recognition (face embeddings)
- Image retrieval (content-based search)
- Person re-identification
- Product similarity

**Configuration**:
```python
from models.components.head_configuration import create_embedding_head

head = create_embedding_head(
    name="face_embedding",
    embed_dim=256,
    projection_layers=[512, 256],  # Optional MLP projection
    use_bn=True,  # Batch normalization before L2 norm
    dropout_rate=0.1
)
```

**Training**:
- **Loss**: `CosineSimilarity`, `TripletLoss`, or custom metric learning losses
- **Metrics**: `cosine_similarity`
- **Label format**: Identity labels for supervised learning, or pairs/triplets for metric learning

**Output**: L2-normalized vectors suitable for cosine similarity computation.

### 5. Ordinal Regression (`ordinal`)

**Purpose**: Ordered categorical classification where class order matters (rating scales, severity levels).

**Architecture**: GlobalAveragePooling2D → Dropout → Dense(num_classes-1, activation='sigmoid') with ordered bias initialization

**Method**: CORAL (Consistent Rank Logits) - uses K-1 binary threshold classifiers for K ordinal classes.

**Use Cases**:
- Age groups (child, teen, adult, senior)
- Damage severity (none, mild, moderate, severe)
- Quality ratings (1-star to 5-star)
- Disease staging

**Configuration**:
```python
from models.components.head_configuration import create_ordinal_head

head = create_ordinal_head(
    name="age_group",
    num_classes=5,  # 5 ordinal classes
    threshold_init="ascending",  # Initialize thresholds in order
    dropout_rate=0.2
)
```

**Training**:
- **Loss**: `OrdinalCrossEntropy` (custom loss for CORAL)
- **Metrics**: `ordinal_accuracy`, `ordinal_mae`
- **Label format**: Integer ordinal class indices (0, 1, 2, ...)

**Prediction**: Predicted class = sum(sigmoid_outputs > 0.5)

## Using Head Types

### Creating Models with Different Head Types

```python
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.components.head_configuration import *
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture

# Create different head types
heads = [
    create_classification_head("object_type", num_classes=10),
    create_multilabel_head("attributes", num_labels=5),
    create_regression_head("age", n_outputs=1, output_scale=100),
    create_embedding_head("identity", embed_dim=128),
    create_ordinal_head("quality", num_classes=4)
]

config = MultiHeadModelConfig(
    input_shape=(224, 224, 3),
    head_configs=heads,
    arch_params={'alpha': 0.75, 'use_pretrained': True}
)

arch = MultiHeadMobileNetV3QATArchitecture(config)
model = arch.get_model()
```

### Automatic Loss and Metrics Selection

```python
from src.utils import get_losses_for_heads, get_metrics_for_heads

# Get appropriate losses for each head type
losses = get_losses_for_heads(heads, from_logits=True)

# Get suitable metrics for each head type
metrics = get_metrics_for_heads(heads)

# Compile model
model.compile(
    optimizer='adam',
    loss=losses,
    metrics=metrics
)
```

### Custom Loss Overrides

```python
# Override default losses for specific heads
loss_overrides = {
    'age': {'loss_type': 'mae'},  # Use MAE instead of MSE for regression
    'identity': tf.keras.losses.TripletSemiHardLoss()  # Custom loss
}

losses = get_losses_for_heads(heads, loss_overrides=loss_overrides)
```

## Extending with Custom Head Types

You can register custom head builders using the `@register_head` decorator:

```python
from models.components.head_builders import register_head
from tensorflow.keras import layers

@register_head("attention_pooling")
def build_attention_pooling_head(head_config, backbone_output):
    # Custom attention-based pooling instead of GAP
    B, H, W, C = backbone_output.shape
    
    # Attention mechanism
    attention = layers.Conv2D(1, 1, activation='sigmoid')(backbone_output)
    attended = backbone_output * attention
    
    # Weighted average pooling
    pooled = layers.GlobalAveragePooling2D()(attended)
    
    if head_config.dropout_rate > 0:
        pooled = layers.Dropout(head_config.dropout_rate)(pooled)
    
    return layers.Dense(
        head_config.num_classes,
        activation=head_config.activation
    )(pooled)
```

Then use it in head configuration:

```python
head = HeadConfiguration(
    name="attended_classifier",
    num_classes=100,
    head_type="attention_pooling",
    dropout_rate=0.2
)
```

## Best Practices

1. **Choose the right head type**: Match the head type to your task requirements
2. **Proper activation**: Use 'linear' for Vela compatibility, apply softmax in post-processing
3. **Loss function alignment**: Use `get_losses_for_heads()` for automatic selection
4. **Data format**: Ensure your training data matches the expected format for each head type
5. **Balanced learning**: Consider loss weights when combining different head types
6. **Regularization**: Adjust dropout rates based on head complexity and data size

## Performance Considerations

- **Standard classification**: Most efficient, well-optimized in TFLite
- **Multi-label**: Similar to classification, sigmoid is efficient
- **Regression**: Lightweight, no softmax needed
- **Embedding**: L2 normalization adds minimal overhead
- **Ordinal**: Slightly more complex due to multiple sigmoid outputs

## Quantization Compatibility

All head types are compatible with:
- **Int8 quantization**: Full post-training quantization
- **Mixed precision**: Int8 backbone + FP16/FP32 heads via `--export-separate-tflite`
- **Vela compiler**: Use 'linear' activation for NPU compatibility

## Examples

See the following files for complete examples:
- `examples/12_alternative_heads.sh` - CLI examples for each head type
- `examples/demo_alternative_heads.py` - Python training examples
- `docs/training_guide.md` - Integration with training workflows