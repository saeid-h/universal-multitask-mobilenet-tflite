"""
Loss functions and utilities for multi-head models.

This module provides loss function helpers that automatically select
appropriate losses based on head types and configurations.
"""

from typing import Dict, Any, Optional, Union
import tensorflow as tf
from models.components.head_configuration import HeadConfiguration


def get_loss_for_head_type(head_type: str, from_logits: bool = True, **kwargs) -> tf.keras.losses.Loss:
    """Get appropriate loss function for a head type.
    
    Args:
        head_type: Type of head ('standard', 'multilabel', 'regression', etc.)
        from_logits: Whether the model outputs logits (True) or probabilities (False)
        **kwargs: Additional arguments passed to the loss function
        
    Returns:
        Configured loss function
        
    Raises:
        ValueError: If head_type is not supported
    """
    if head_type == "standard":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, **kwargs)
    
    elif head_type == "multilabel":
        return tf.keras.losses.BinaryCrossentropy(from_logits=False, **kwargs)  # sigmoid outputs
    
    elif head_type == "regression":
        # Default to MSE, but allow override
        loss_type = kwargs.pop('loss_type', 'mse')
        if loss_type == 'mse':
            return tf.keras.losses.MeanSquaredError(**kwargs)
        elif loss_type == 'mae':
            return tf.keras.losses.MeanAbsoluteError(**kwargs)
        elif loss_type == 'huber':
            return tf.keras.losses.Huber(**kwargs)
        else:
            raise ValueError(f"Unknown regression loss type: {loss_type}")
    
    elif head_type == "embedding":
        # Default to triplet loss, but allow override
        loss_type = kwargs.pop('loss_type', 'cosine')
        if loss_type == 'cosine':
            return tf.keras.losses.CosineSimilarity(**kwargs)
        elif loss_type == 'mse':
            return tf.keras.losses.MeanSquaredError(**kwargs)  # For reconstruction tasks
        else:
            raise ValueError(f"Unknown embedding loss type: {loss_type}")
    
    elif head_type == "ordinal":
        # Use binary crossentropy for each threshold
        return tf.keras.losses.BinaryCrossentropy(from_logits=False, **kwargs)  # sigmoid outputs
    
    elif head_type == "ssd_detection":
        # SSD uses combined classification + localization loss
        # Return a custom loss that handles both outputs
        return SSDLoss(**kwargs)
    
    elif head_type == "yolo_detection":
        # YOLO uses combined objectness + classification + localization loss
        return YOLOLoss(**kwargs)
    
    elif head_type == "segmentation":
        # Pixel-wise classification
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits, **kwargs)
    
    elif head_type == "keypoint_detection":
        # Heatmap regression (MSE on heatmap values)
        return tf.keras.losses.MeanSquaredError(**kwargs)
    
    else:
        raise ValueError(f"Unknown head type: {head_type}")


def get_losses_for_heads(head_configs: list, from_logits: bool = True, 
                        loss_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, tf.keras.losses.Loss]:
    """Get loss functions for a list of head configurations.
    
    Args:
        head_configs: List of HeadConfiguration objects
        from_logits: Whether model outputs logits (True) or probabilities (False)
        loss_overrides: Optional dictionary to override default losses per head name
        
    Returns:
        Dictionary mapping head names to loss functions
    """
    losses = {}
    loss_overrides = loss_overrides or {}
    
    for head_config in head_configs:
        head_name = head_config.name
        
        if head_name in loss_overrides:
            # Use override
            loss_config = loss_overrides[head_name]
            if isinstance(loss_config, tf.keras.losses.Loss):
                losses[head_name] = loss_config
            elif isinstance(loss_config, dict):
                losses[head_name] = get_loss_for_head_type(head_config.head_type, from_logits, **loss_config)
            else:
                raise ValueError(f"Invalid loss override for head '{head_name}': {loss_config}")
        else:
            # Use default for head type
            losses[head_name] = get_loss_for_head_type(head_config.head_type, from_logits)
    
    return losses


def get_metrics_for_head_type(head_type: str) -> list:
    """Get appropriate metrics for a head type.
    
    Args:
        head_type: Type of head
        
    Returns:
        List of metric names or metric instances
    """
    if head_type == "standard":
        return ['accuracy', 'sparse_top_k_categorical_accuracy']
    
    elif head_type == "multilabel":
        return ['binary_accuracy', 'precision', 'recall']
    
    elif head_type == "regression":
        return ['mae', 'mse']
    
    elif head_type == "embedding":
        return ['cosine_similarity']
    
    elif head_type == "ordinal":
        return ['binary_accuracy', 'mae']  # MAE on ordinal predictions
    
    elif head_type == "ssd_detection":
        return ['binary_accuracy']  # For classification component
    
    elif head_type == "yolo_detection":
        return ['binary_accuracy']  # For objectness component
    
    elif head_type == "segmentation":
        return ['accuracy', 'sparse_categorical_accuracy']
    
    elif head_type == "keypoint_detection":
        return ['mse', 'mae']  # Heatmap regression metrics
    
    else:
        return ['accuracy']  # Default fallback


def get_metrics_for_heads(head_configs: list, 
                         metric_overrides: Optional[Dict[str, list]] = None) -> Dict[str, list]:
    """Get metrics for a list of head configurations.
    
    Args:
        head_configs: List of HeadConfiguration objects
        metric_overrides: Optional dictionary to override default metrics per head name
        
    Returns:
        Dictionary mapping head names to lists of metrics
    """
    metrics = {}
    metric_overrides = metric_overrides or {}
    
    for head_config in head_configs:
        head_name = head_config.name
        
        if head_name in metric_overrides:
            metrics[head_name] = metric_overrides[head_name]
        else:
            metrics[head_name] = get_metrics_for_head_type(head_config.head_type)
    
    return metrics


class OrdinalCrossEntropy(tf.keras.losses.Loss):
    """Cross-entropy loss for ordinal regression using CORAL encoding.
    
    Computes binary cross-entropy for each threshold in ordinal regression.
    """
    
    def __init__(self, num_classes: int, name: str = "ordinal_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
    
    def call(self, y_true, y_pred):
        """Compute ordinal cross-entropy loss.
        
        Args:
            y_true: True ordinal labels (shape: [batch])
            y_pred: Predicted threshold probabilities (shape: [batch, num_thresholds])
            
        Returns:
            Loss tensor
        """
        # Convert ordinal labels to binary threshold targets
        y_true = tf.cast(y_true, tf.int32)
        batch_size = tf.shape(y_true)[0]
        
        # Create binary targets for each threshold
        # If true class is k, then thresholds 0..k-1 should be 1, k..num_thresholds-1 should be 0
        threshold_indices = tf.range(self.num_thresholds, dtype=tf.int32)  # [0, 1, 2, ...]
        threshold_indices = tf.expand_dims(threshold_indices, 0)  # [1, num_thresholds]
        y_true_expanded = tf.expand_dims(y_true, 1)  # [batch, 1]
        
        # Binary targets: 1 if threshold_index < y_true, 0 otherwise
        binary_targets = tf.cast(threshold_indices < y_true_expanded, tf.float32)
        
        # Compute binary cross-entropy for each threshold
        bce = tf.keras.losses.binary_crossentropy(binary_targets, y_pred, from_logits=False)
        
        return tf.reduce_mean(bce)
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes})
        return config


def ordinal_accuracy(y_true, y_pred):
    """Accuracy metric for ordinal regression.
    
    Converts threshold probabilities back to ordinal predictions and computes accuracy.
    
    Args:
        y_true: True ordinal labels
        y_pred: Predicted threshold probabilities
        
    Returns:
        Accuracy scalar
    """
    # Convert threshold probabilities to ordinal predictions
    # Predicted class = sum(sigmoid_outputs > 0.5)
    binary_preds = tf.cast(y_pred > 0.5, tf.float32)
    ordinal_preds = tf.reduce_sum(binary_preds, axis=1)
    
    # Compute accuracy
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(tf.cast(tf.equal(ordinal_preds, y_true), tf.float32))


def ordinal_mae(y_true, y_pred):
    """Mean absolute error for ordinal regression.
    
    Args:
        y_true: True ordinal labels
        y_pred: Predicted threshold probabilities
        
    Returns:
        MAE scalar
    """
    # Convert to ordinal predictions
    binary_preds = tf.cast(y_pred > 0.5, tf.float32)
    ordinal_preds = tf.reduce_sum(binary_preds, axis=1)
    
    # Compute MAE
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(tf.abs(ordinal_preds - y_true))


class SSDLoss(tf.keras.losses.Loss):
    """Combined loss for SSD detection (classification + localization).
    
    Combines binary/multi-class crossentropy for classification with
    smooth L1 loss for bounding box regression.
    """
    
    def __init__(self, alpha: float = 1.0, neg_pos_ratio: float = 3.0, 
                 name: str = "ssd_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha  # Weight for localization loss
        self.neg_pos_ratio = neg_pos_ratio  # Negative to positive ratio for hard negative mining
    
    def call(self, y_true, y_pred):
        """Compute SSD loss.
        
        Args:
            y_true: Dictionary with 'boxes' and 'labels' ground truth
            y_pred: Dictionary with 'boxes' and 'scores' predictions
            
        Returns:
            Combined loss scalar
        """
        # Extract predictions and targets
        pred_boxes = y_pred['boxes']  # [batch, num_anchors, 4]
        pred_scores = y_pred['scores']  # [batch, num_anchors, num_classes]
        
        true_boxes = y_true['boxes']  # [batch, num_anchors, 4] 
        true_labels = y_true['labels']  # [batch, num_anchors] (0 = background)
        
        # Classification loss (with hard negative mining)
        classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
            true_labels, pred_scores, from_logits=False
        )
        
        # Localization loss (only for positive anchors)
        positive_mask = tf.cast(true_labels > 0, tf.float32)  # Ignore background
        
        # Smooth L1 loss for bounding boxes
        box_diff = pred_boxes - true_boxes
        abs_diff = tf.abs(box_diff)
        smooth_l1 = tf.where(
            abs_diff < 1.0,
            0.5 * tf.square(box_diff),
            abs_diff - 0.5
        )
        localization_loss = tf.reduce_sum(smooth_l1 * tf.expand_dims(positive_mask, -1), axis=-1)
        
        # Normalize by number of positive anchors
        num_positives = tf.reduce_sum(positive_mask, axis=1, keepdims=True)
        num_positives = tf.maximum(num_positives, 1.0)  # Avoid division by zero
        
        classification_loss = tf.reduce_sum(classification_loss * positive_mask, axis=1) / num_positives[:, 0]
        localization_loss = tf.reduce_sum(localization_loss, axis=1) / num_positives[:, 0]
        
        # Combined loss
        total_loss = classification_loss + self.alpha * localization_loss
        return tf.reduce_mean(total_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'neg_pos_ratio': self.neg_pos_ratio
        })
        return config


class YOLOLoss(tf.keras.losses.Loss):
    """YOLO detection loss (objectness + classification + localization).
    
    Computes the YOLO loss combining objectness confidence, 
    class probabilities, and bounding box coordinates.
    """
    
    def __init__(self, lambda_coord: float = 5.0, lambda_noobj: float = 0.5,
                 name: str = "yolo_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_coord = lambda_coord  # Weight for coordinate loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object loss
    
    def call(self, y_true, y_pred):
        """Compute YOLO loss.
        
        Args:
            y_true: Ground truth tensor [batch, grid_h, grid_w, num_boxes * (5 + num_classes)]
            y_pred: Predicted tensor [batch, grid_h, grid_w, num_boxes * (5 + num_classes)]
            
        Returns:
            YOLO loss scalar
        """
        # This is a simplified version - full YOLO loss is more complex
        # For production, use tf.keras.losses.binary_crossentropy for components
        
        # Split predictions into components
        # Format: [x, y, w, h, objectness, class1, class2, ...]
        
        # Objectness loss (binary crossentropy)
        obj_loss = tf.keras.losses.binary_crossentropy(
            y_true[..., 4::5+1], y_pred[..., 4::5+1], from_logits=True  # Simplified indexing
        )
        
        # Coordinate loss (MSE for present objects)
        coord_loss = tf.keras.losses.mse(y_true[..., :4], y_pred[..., :4])
        
        # Class loss (categorical crossentropy for present objects)
        class_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true[..., 5:], y_pred[..., 5:], from_logits=True
        )
        
        # Combine losses (simplified weighting)
        total_loss = (self.lambda_coord * coord_loss + 
                     obj_loss + 
                     class_loss)
        
        return tf.reduce_mean(total_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lambda_coord': self.lambda_coord,
            'lambda_noobj': self.lambda_noobj
        })
        return config