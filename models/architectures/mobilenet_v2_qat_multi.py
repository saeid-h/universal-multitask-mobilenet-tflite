"""
Multi-head MobileNetV2 QAT-optimized architecture implementation.

This module implements multi-head MobileNetV2 with QAT optimization using
Keras applications, supporting multiple task heads sharing a single backbone.

Key features:
- Direct use of Keras MobileNetV2 with ImageNet pre-trained weights (RGB only)
- Configurable alpha values (0.35, 0.50, 0.75, 1.0, 1.3, 1.4) - Keras supported
- QAT-compatible architecture design
- Multiple task heads with shared backbone (any head type from the registry)
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

from ..factory import ModelArchitectureFactory
from .multi_head_base import MultiHeadMobileNetArchitecture


_ALPHA_NAME = {
    0.35: '0_35',
    0.50: '0_50',
    0.75: '0_75',
    1.0: '1_0',
    1.3: '1_3',
    1.4: '1_4',
}


class MultiHeadMobileNetV2QATArchitecture(MultiHeadMobileNetArchitecture):
    """Multi-head MobileNetV2 QAT-optimized architecture implementation.

    Supports alpha values: 0.35, 0.50, 0.75, 1.0, 1.3, 1.4 (Keras MobileNetV2
    supported values). All alpha values support ImageNet pretrained weights
    when input is RGB.
    """

    @property
    def name(self) -> str:
        alpha = self.config.arch_params.get('alpha', 0.35)
        alpha_str = _ALPHA_NAME.get(alpha, str(alpha).replace('.', '_'))
        head_classes = [head.num_classes for head in self.head_configs]
        head_str = '_'.join(map(str, head_classes))
        return f"mobilenet_v2_qat_multi_{alpha_str}_{head_str}"

    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        # Mirror the single-head V2 QAT supported set (mobilenet_v2_qat.py:63-74)
        return [
            (96, 96, 1),
            (96, 96, 3),
            (128, 128, 1),
            (128, 128, 3),
            (160, 160, 1),
            (160, 160, 3),
            (224, 224, 1),
            (224, 224, 3),
            (256, 256, 1),
            (256, 256, 3),
        ]

    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        alpha = self.config.arch_params.get('alpha', 0.35)
        # Reference values from mobilenet_v2_qat.py:83-90
        reference_params = {
            0.35: 250_000,
            0.50: 500_000,
            0.75: 1_100_000,
            1.0: 2_200_000,
            1.3: 3_800_000,
            1.4: 4_400_000,
        }
        base_params = reference_params.get(alpha, 250_000)
        head_params = sum(head.num_classes * 1000 for head in self.head_configs)
        # Allow 15% tolerance for the backbone, then add head params.
        min_params = int(base_params * 0.85) + head_params
        max_params = int(base_params * 1.15) + head_params
        return (min_params, max_params)

    def validate_config(self) -> None:
        super().validate_config()

        alpha = self.config.arch_params.get('alpha', 0.35)
        valid_alphas = [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]
        if alpha not in valid_alphas:
            raise ValueError(
                f"alpha must be one of {valid_alphas} (Keras MobileNetV2 "
                f"supported values), got {alpha}"
            )

        self.validate_input_shape(self.config.input_shape)

        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"MobileNetV2 requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV2 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )

    def build_backbone(self) -> tf.keras.Model:
        alpha = self.config.arch_params.get('alpha', 0.35)
        use_pretrained = self.config.arch_params.get('use_pretrained', True)
        input_shape = self.config.input_shape
        _, _, channels = input_shape

        can_use_imagenet = use_pretrained and channels == 3
        weights = 'imagenet' if can_use_imagenet else None

        backbone = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
            alpha=alpha,
            pooling=None,
        )

        if can_use_imagenet:
            backbone.trainable = False

        return backbone


def create_mobilenet_v2_qat_multi_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for multi-head MobileNetV2 QAT variants."""
    configs = []

    alpha_configs = [
        (0.35, "Ultra-lightweight MCU (~250KB)"),
        (0.50, "Lightweight MCU (~500KB)"),
        (0.75, "Balanced mobile (~1.1MB)"),
        (1.0, "Standard mobile (~2.2MB)"),
        (1.3, "Large mobile (~3.8MB)"),
        (1.4, "X-large mobile (~4.4MB)"),
    ]
    head_configs = [
        ([2], "Single head: Person detection"),
        ([2, 2], "Two heads: Person + Gender"),
        ([2, 2, 5], "Three heads: Person + Gender + Age"),
        ([5, 2], "Two heads: 5-class + 2-class"),
    ]

    for alpha, alpha_description in alpha_configs:
        for head_list, head_description in head_configs:
            alpha_str = _ALPHA_NAME.get(alpha, str(alpha).replace('.', '_'))
            head_str = '_'.join(map(str, head_list))
            arch_name = f"mobilenet_v2_qat_multi_{alpha_str}_{head_str}"

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
                    f"MobileNetV2 QAT Multi-head with alpha={alpha}, "
                    f"heads={head_list} - {alpha_description}, {head_description}"
                ),
            }
            configs.append((arch_name, default_config))

    return configs


def _register_mobilenet_v2_qat_multi_variants():
    """Register all multi-head MobileNetV2 QAT variants with the factory."""
    for arch_name, default_config in create_mobilenet_v2_qat_multi_configs():
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MultiHeadMobileNetV2QATArchitecture,
            default_config=default_config,
        )


_register_mobilenet_v2_qat_multi_variants()
