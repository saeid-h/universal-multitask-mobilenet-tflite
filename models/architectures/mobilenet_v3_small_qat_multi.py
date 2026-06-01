"""
Multi-head architecture for the project's custom MobileNetV3-Small backbone.

The Keras-applications MobileNetV3Small (used by the `v3` backbone) accepts
only its specific set of input shapes and disallows grayscale ImageNet
weights. This module wraps the project's own custom MobileNetV3-Small
implementation (see mobilenet_v3_small.py) into the multi-head pipeline so
projects that need grayscale inputs or non-standard resolutions can target
a V3-Small architecture too.

Key differences vs. --backbone v3 (Keras applications):
- Wider supported_input_shapes (grayscale variants at 96 / 128 / 224 / 256).
- No ImageNet pretrained weights bundled — --use-pretrained is ignored.
- Identical multi-head plumbing (stride taps, head builders) inherited
  from MultiHeadMobileNetArchitecture.
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf

from ..base import ModelConfig
from ..factory import ModelArchitectureFactory
from ..components import (
    InvertedResidualBlock,
    StandardConvBlock,
    apply_activation,
    ComponentConfig,
)
from .multi_head_base import MultiHeadMobileNetArchitecture
from .mobilenet_v3_small import MobileNetV3Small as _CustomV3SmallSingleHead
from ._keras_app_taps import features_by_stride_from_keras_model


_ALPHA_NAME = {0.25: '0_25', 0.5: '0_50', 0.75: '0_75', 1.0: '1_0'}


class MultiHeadMobileNetV3SmallCustomQATArchitecture(MultiHeadMobileNetArchitecture):
    """Multi-head wrapper for the project's custom MobileNetV3-Small backbone.

    Width multipliers: 0.25, 0.5, 0.75, 1.0 (mirrors the single-head sibling).
    No ImageNet pretrained weights; --use-pretrained is silently ignored.
    """

    @property
    def name(self) -> str:
        alpha = self.config.arch_params.get('alpha', 0.25)
        alpha_str = _ALPHA_NAME.get(alpha, str(alpha).replace('.', '_'))
        head_classes = [head.num_classes for head in self.head_configs]
        head_str = '_'.join(map(str, head_classes))
        return f"mobilenet_v3_small_custom_qat_multi_{alpha_str}_{head_str}"

    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        # Mirror the single-head custom V3-Small sibling.
        return [
            (96, 96, 1),
            (128, 128, 1),
            (224, 224, 1),
            (256, 256, 1),
            (96, 96, 3),
            (128, 128, 3),
            (224, 224, 3),
            (256, 256, 3),
        ]

    @property
    def parameter_count_range(self) -> Tuple[int, int]:
        # The single-head custom V3 uses a base range of 50K..3M; the multi-
        # head variant is essentially the same backbone size plus per-head
        # parameters (~1000 per class).
        head_params = sum(head.num_classes * 1000 for head in self.head_configs)
        return (40_000 + head_params, 3_000_000 + head_params)

    def validate_config(self) -> None:
        super().validate_config()

        alpha = self.config.arch_params.get('alpha', 0.25)
        valid_alphas = [0.25, 0.5, 0.75, 1.0]
        if alpha not in valid_alphas:
            raise ValueError(
                f"alpha must be one of {valid_alphas} (custom V3-Small "
                f"supported values), got {alpha}"
            )

        self.validate_input_shape(self.config.input_shape)

        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"Custom V3-Small requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        if channels not in [1, 3]:
            raise ValueError(
                f"Custom V3-Small supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )

        # No pretrained weights available; we silently ignore the flag.

    def build_backbone(self) -> tf.keras.Model:
        """Build the custom V3-Small backbone (feature extractor only).

        Mirrors mobilenet_v3_small.py:180-249 (initial conv -> IR blocks ->
        final 1x1 conv -> BN -> hard_swish), stopping before the GAP /
        Dense / classification head so the multi-head base can attach
        configured heads on top.
        """
        alpha = self.config.arch_params.get('alpha', 0.25)

        # Build a single-head V3-Small helper just to reuse its layer specs,
        # channel rounding, and width-multiplier-adjusted block construction.
        v3_arch_params = dict(self.config.arch_params)
        v3_arch_params['width_multiplier'] = alpha
        v3_arch_params.setdefault('architecture_type', 'mobilenet_v3_small')
        v3_config = ModelConfig(
            input_shape=self.config.input_shape,
            num_classes=2,  # unused; we skip the head
            arch_params=v3_arch_params,
            weight_decay=self.config.weight_decay,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate,
        )
        v3_helper = _CustomV3SmallSingleHead(v3_config)

        inputs = tf.keras.layers.Input(
            shape=self.config.input_shape,
            name='v3small_backbone_input',
        )

        initial_channels = v3_helper._apply_width_multiplier(16)
        initial_conv_config = ComponentConfig(
            activation='hard_swish',
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False,
        )
        x = StandardConvBlock(
            config=initial_conv_config,
            filters=initial_channels,
            kernel_size=3,
            strides=2,
        ).call(inputs, training=False)

        for input_ch, exp_ratio, output_ch, use_se, activation, stride, kernel_size in v3_helper.LAYER_SPECS:
            output_channels = v3_helper._apply_width_multiplier(output_ch)

            if v3_helper.width_multiplier <= 0.25:
                exp_ratio = max(1, exp_ratio * 0.5)
            elif v3_helper.width_multiplier <= 0.5:
                exp_ratio = max(1, exp_ratio * 0.75)

            block = InvertedResidualBlock(
                config=self.config,
                output_channels=output_channels,
                expansion_ratio=int(exp_ratio),
                stride=stride,
                use_se=use_se,
                se_reduction_ratio=4,
                activation=activation,
                kernel_size=kernel_size,
            )
            x = block.call(x, training=False)

        # Final 1x1 conv + BN + hard_swish (no GAP/Dense — that lives in heads).
        final_channels = v3_helper._apply_width_multiplier(576)
        x = tf.keras.layers.Conv2D(
            final_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            name='v3small_backbone_final_conv',
        )(x)
        x = tf.keras.layers.BatchNormalization(
            momentum=self.config.batch_norm_momentum,
            epsilon=self.config.batch_norm_epsilon,
            name='v3small_backbone_final_bn',
        )(x, training=False)
        x = apply_activation(x, 'hard_swish', training=False)

        return tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name=f"mobilenet_v3_small_custom_backbone_{_ALPHA_NAME.get(alpha, alpha)}",
        )

    def _features_by_stride(
        self,
        backbone: tf.keras.Model,
        input_tensor: tf.Tensor,
        backbone_output: tf.Tensor,
    ) -> Dict[int, tf.Tensor]:
        """Expose stride 4/8/16/32 feature maps from the custom V3-Small."""
        return features_by_stride_from_keras_model(
            backbone,
            input_tensor,
            self.config.input_shape,
        )


def create_mobilenet_v3_small_custom_qat_multi_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for multi-head custom V3-Small variants."""
    configs = []
    alpha_configs = [
        (0.25, "Ultra-lightweight MCU"),
        (0.5,  "Lightweight MCU"),
        (0.75, "Balanced mobile"),
        (1.0,  "Standard mobile"),
    ]
    head_configs = [
        ([2], "Single head: Person detection"),
        ([2, 2], "Two heads: Person + Gender"),
        ([2, 2, 5], "Three heads"),
    ]

    for alpha, alpha_desc in alpha_configs:
        for head_list, head_desc in head_configs:
            alpha_str = _ALPHA_NAME.get(alpha, str(alpha).replace('.', '_'))
            head_str = '_'.join(map(str, head_list))
            arch_name = f"mobilenet_v3_small_custom_qat_multi_{alpha_str}_{head_str}"

            head_configs_list = [
                {
                    'name': f"head_{i+1}",
                    'num_classes': n,
                    'activation': 'softmax',
                    'dropout_rate': 0.2,
                    'loss_weight': 1.0,
                }
                for i, n in enumerate(head_list)
            ]

            configs.append((arch_name, {
                'input_shape': (96, 96, 1),
                'num_classes': 2,
                'arch_params': {
                    'alpha': alpha,
                    'use_pretrained': False,  # not supported for custom V3
                },
                'head_configs': head_configs_list,
                'training_mode': 'joint',
                'inference_mode': 'all_active',
                'description': (
                    f"Custom V3-Small Multi-head with alpha={alpha}, "
                    f"heads={head_list} - {alpha_desc}, {head_desc}"
                ),
            }))

    return configs


def _register_mobilenet_v3_small_custom_qat_multi_variants():
    """Register all custom V3-Small multi-head variants with the factory."""
    for arch_name, default_config in create_mobilenet_v3_small_custom_qat_multi_configs():
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MultiHeadMobileNetV3SmallCustomQATArchitecture,
            default_config=default_config,
        )


_register_mobilenet_v3_small_custom_qat_multi_variants()
