"""
Multi-head MobileNetV1 QAT-optimized architecture implementation.

This module implements multi-head MobileNetV1 with QAT optimization using
Keras applications, supporting multiple task heads sharing a single backbone.

Key features:
- Direct use of Keras MobileNetV1 with ImageNet pre-trained weights (RGB only)
- Configurable alpha values (0.25, 0.50, 0.75, 1.0) - Keras supported
- QAT-compatible architecture design
- Multiple task heads with shared backbone (any head type from the registry)
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras.applications import MobileNet

from ..factory import ModelArchitectureFactory
from .multi_head_base import MultiHeadMobileNetArchitecture
from ._keras_app_taps import features_by_stride_from_keras_model


class MultiHeadMobileNetV1QATArchitecture(MultiHeadMobileNetArchitecture):
    """Multi-head MobileNetV1 QAT-optimized architecture implementation.

    Supports alpha values: 0.25, 0.50, 0.75, 1.0 (Keras MobileNetV1 supported).
    All alpha values support ImageNet pretrained weights when input is RGB.
    """

    @property
    def name(self) -> str:
        alpha = self.config.arch_params.get('alpha', 0.25)
        alpha_mapping = {
            0.25: '0_25',
            0.50: '0_50',
            0.75: '0_75',
            1.0: '1_0',
        }
        alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
        head_classes = [head.num_classes for head in self.head_configs]
        head_str = '_'.join(map(str, head_classes))
        return f"mobilenet_v1_qat_multi_{alpha_str}_{head_str}"

    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        # Mirror the single-head V1 QAT supported set (mobilenet_v1_qat.py:51-60)
        return [
            (96, 96, 1),
            (128, 128, 1),
            (160, 160, 1),
            (224, 224, 1),
            (96, 96, 3),
            (128, 128, 3),
            (160, 160, 3),
            (224, 224, 3),
        ]

    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        alpha = self.config.arch_params.get('alpha', 0.25)
        # Backbone-only ranges from mobilenet_v1_qat.py:69-74
        alpha_to_params = {
            0.25: (200_000, 300_000),
            0.50: (800_000, 1_200_000),
            0.75: (1_800_000, 2_700_000),
            1.0: (3_200_000, 4_800_000),
        }
        base_range = alpha_to_params.get(alpha, (100_000, 5_000_000))
        # Add head parameters (rough estimate: ~1000 params per class per head)
        head_params = sum(head.num_classes * 1000 for head in self.head_configs)
        return (base_range[0] + head_params, base_range[1] + head_params)

    def validate_config(self) -> None:
        super().validate_config()

        alpha = self.config.arch_params.get('alpha', 0.25)
        valid_alphas = [0.25, 0.50, 0.75, 1.0]
        if alpha not in valid_alphas:
            raise ValueError(
                f"alpha must be one of {valid_alphas} (Keras MobileNetV1 "
                f"supported values), got {alpha}"
            )

        self.validate_input_shape(self.config.input_shape)

        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"MobileNetV1 requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV1 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )

    def build_backbone(self) -> tf.keras.Model:
        alpha = self.config.arch_params.get('alpha', 0.25)
        use_pretrained = self.config.arch_params.get('use_pretrained', True)
        input_shape = self.config.input_shape
        _, _, channels = input_shape

        # ImageNet weights are only available for RGB inputs
        can_use_imagenet = use_pretrained and channels == 3
        weights = 'imagenet' if can_use_imagenet else None

        backbone = MobileNet(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            alpha=alpha,
            pooling=None,
        )

        # Mirror the single-head behavior: freeze backbone when using ImageNet
        if can_use_imagenet:
            backbone.trainable = False

        return backbone

    def _features_by_stride(
        self,
        backbone: tf.keras.Model,
        input_tensor: tf.Tensor,
        backbone_output: tf.Tensor,
    ) -> Dict[int, tf.Tensor]:
        """Expose stride-4/8/16/32 feature maps from MobileNetV1."""
        return features_by_stride_from_keras_model(
            backbone,
            input_tensor,
            self.config.input_shape,
        )


def create_mobilenet_v1_qat_multi_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for multi-head MobileNetV1 QAT variants."""
    configs = []

    alpha_configs = [
        (0.25, "Ultra-lightweight MCU (~200KB)"),
        (0.50, "Lightweight MCU (~800KB)"),
        (0.75, "Balanced mobile (~2.2MB)"),
        (1.0, "Standard mobile (~4.2MB)"),
    ]
    head_configs = [
        ([2], "Single head: Person detection"),
        ([2, 2], "Two heads: Person + Gender"),
        ([2, 2, 5], "Three heads: Person + Gender + Age"),
        ([5, 2], "Two heads: 5-class + 2-class"),
    ]

    alpha_mapping = {0.25: '0_25', 0.50: '0_50', 0.75: '0_75', 1.0: '1_0'}

    for alpha, alpha_description in alpha_configs:
        for head_list, head_description in head_configs:
            alpha_str = alpha_mapping.get(alpha, str(alpha).replace('.', '_'))
            head_str = '_'.join(map(str, head_list))
            arch_name = f"mobilenet_v1_qat_multi_{alpha_str}_{head_str}"

            head_configs_list = []
            for i, num_classes in enumerate(head_list):
                head_configs_list.append({
                    'name': f"head_{i+1}",
                    'num_classes': num_classes,
                    'activation': 'softmax',
                    'dropout_rate': 0.2,
                    'loss_weight': 1.0,
                })

            default_config = {
                'input_shape': (224, 224, 3),
                'num_classes': 2,
                'arch_params': {
                    'alpha': alpha,
                    'use_pretrained': True,
                },
                'head_configs': head_configs_list,
                'training_mode': 'joint',
                'inference_mode': 'all_active',
                'description': (
                    f"MobileNetV1 QAT Multi-head with alpha={alpha}, "
                    f"heads={head_list} - {alpha_description}, {head_description}"
                ),
            }
            configs.append((arch_name, default_config))

    return configs


def _register_mobilenet_v1_qat_multi_variants():
    """Register all multi-head MobileNetV1 QAT variants with the factory."""
    for arch_name, default_config in create_mobilenet_v1_qat_multi_configs():
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MultiHeadMobileNetV1QATArchitecture,
            default_config=default_config,
        )


_register_mobilenet_v1_qat_multi_variants()
