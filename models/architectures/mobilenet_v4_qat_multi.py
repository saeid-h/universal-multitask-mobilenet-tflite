"""
Multi-head MobileNetV4 QAT-optimized architecture implementation.

This module implements multi-head MobileNetV4-Conv-S with QAT support,
supporting multiple task heads sharing a single backbone.

Unlike V1/V2/V3, MobileNetV4 has no entry in tf.keras.applications, so this
class builds the backbone directly from the project's custom UIB-based V4
implementation (see models/architectures/mobilenet_v4.py). The forward
pass mirrors MobileNetV4ConvS.build_model() up to and including the final
1x1 conv, then returns that as a standalone backbone Keras model. The
multi-head base attaches task heads on top via the head-builder registry.

Key features:
- Custom UIB-based MobileNetV4-Conv-S backbone (no ImageNet weights available)
- Configurable alpha values (0.25, 0.50, 0.75, 1.0) via width multiplier
- QAT-compatible architecture design
- Multiple task heads with shared backbone (any head type from the registry)
"""

from typing import Dict, Any, Tuple, List
import tensorflow as tf

from ..factory import ModelArchitectureFactory
from ..components import (
    StandardConvBlock,
    ComponentConfig,
)
from .multi_head_base import MultiHeadMobileNetArchitecture
from .mobilenet_v4 import MobileNetV4ConvS
from ._keras_app_taps import features_by_stride_from_keras_model


_ALPHA_NAME = {
    0.25: '0_25',
    0.50: '0_50',
    0.75: '0_75',
    1.0: '1_0',
}


class MultiHeadMobileNetV4QATArchitecture(MultiHeadMobileNetArchitecture):
    """Multi-head MobileNetV4 QAT-optimized architecture implementation.

    Supports alpha values: 0.25, 0.50, 0.75, 1.0 (interpreted as width
    multiplier on the custom MobileNetV4-Conv-S spec).
    """

    @property
    def name(self) -> str:
        alpha = self.config.arch_params.get('alpha', 0.25)
        alpha_str = _ALPHA_NAME.get(alpha, str(alpha).replace('.', '_'))
        head_classes = [head.num_classes for head in self.head_configs]
        head_str = '_'.join(map(str, head_classes))
        return f"mobilenet_v4_qat_multi_{alpha_str}_{head_str}"

    @property
    def supported_input_shapes(self) -> List[Tuple[int, int, int]]:
        # Mirror the single-head V4 QAT supported set (mobilenet_v4_qat.py:51-60)
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
        # Backbone-only ranges from mobilenet_v4_qat.py:70-75
        alpha_to_params = {
            0.25: (200_000, 300_000),
            0.50: (800_000, 1_200_000),
            0.75: (1_800_000, 2_700_000),
            1.0: (3_200_000, 4_800_000),
        }
        base_range = alpha_to_params.get(alpha, (100_000, 5_000_000))
        head_params = sum(head.num_classes * 1000 for head in self.head_configs)
        return (base_range[0] + head_params, base_range[1] + head_params)

    def validate_config(self) -> None:
        super().validate_config()

        alpha = self.config.arch_params.get('alpha', 0.25)
        valid_alphas = [0.25, 0.50, 0.75, 1.0]
        if alpha not in valid_alphas:
            raise ValueError(
                f"alpha must be one of {valid_alphas} (MobileNetV4 supported "
                f"values), got {alpha}"
            )

        self.validate_input_shape(self.config.input_shape)

        height, width, channels = self.config.input_shape
        if height < 32 or width < 32:
            raise ValueError(
                f"MobileNetV4 requires minimum input size of 32x32, "
                f"got {height}x{width}"
            )
        if channels not in [1, 3]:
            raise ValueError(
                f"MobileNetV4 supports 1 (grayscale) or 3 (RGB) channels, "
                f"got {channels}"
            )

        # V4 has no ImageNet weights in this project; warn if the user asked.
        if self.config.arch_params.get('use_pretrained', False):
            # Mirror how V3 handles the no-weights case: just no-op, do not error.
            # The CLI's --use-pretrained is best-effort; V4 simply has nothing to load.
            pass

    def build_backbone(self) -> tf.keras.Model:
        """Build the MobileNetV4-Conv-S feature extractor as a Keras model.

        Replicates MobileNetV4ConvS.build_model() up through the final 1x1
        conv (i.e. the feature map immediately before GAP + classification),
        returning that subgraph as a standalone backbone. The multi-head base
        then attaches task heads on top.
        """
        alpha = self.config.arch_params.get('alpha', 0.25)

        # Stand up a MobileNetV4ConvS helper to reuse its layer specs, channel
        # rounding, and UIB-block factory. We do NOT call its build_model();
        # we only use its forward-pass primitives.
        v4_arch_params = dict(self.config.arch_params)
        v4_arch_params['width_multiplier'] = alpha
        v4_arch_params.setdefault('dropout_rate', 0.2)
        v4_arch_params.setdefault('architecture_type', 'mobilenet_v4_conv_s')

        # Build a ModelConfig compatible with MobileNetV4ConvS (num_classes is
        # ignored here because we skip the classification head).
        from ..base import ModelConfig
        v4_config = ModelConfig(
            input_shape=self.config.input_shape,
            num_classes=2,
            arch_params=v4_arch_params,
            weight_decay=self.config.weight_decay,
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            activation=self.config.activation,
            dropout_rate=self.config.dropout_rate,
        )
        v4_helper = MobileNetV4ConvS(v4_config)

        # Forward pass: copy of mobilenet_v4.py:262-313 up to (and including)
        # the final 1x1 conv, omitting GAP / Dropout / Dense.
        inputs = tf.keras.layers.Input(
            shape=self.config.input_shape,
            name='v4_backbone_input',
        )

        conv_block_config = ComponentConfig(
            activation='relu6',
            batch_norm_momentum=self.config.batch_norm_momentum,
            batch_norm_epsilon=self.config.batch_norm_epsilon,
            use_batch_norm=True,
            use_bias=False,
        )

        initial_channels = v4_helper._make_divisible(32 * v4_helper.width_multiplier)
        x = StandardConvBlock(
            config=conv_block_config,
            filters=initial_channels,
            kernel_size=3,
            strides=2,
        ).call(inputs, training=False)

        for block_type, _, exp_ratio, output_ch, stride, kernel, se_ratio in v4_helper.LAYER_SPECS:
            uib_block = v4_helper._create_uib_block(
                block_type=block_type,
                output_channels=output_ch,
                expansion_ratio=exp_ratio,
                stride=stride,
                kernel_size=kernel,
                se_ratio=se_ratio,
            )
            x = uib_block.call(x, training=False)

        x = StandardConvBlock(
            config=conv_block_config,
            filters=v4_helper.final_conv_channels,
            kernel_size=1,
            strides=1,
        ).call(x, training=False)

        backbone = tf.keras.Model(
            inputs=inputs,
            outputs=x,
            name=f"mobilenet_v4_conv_s_backbone_{_ALPHA_NAME.get(alpha, alpha)}",
        )
        return backbone

    def _features_by_stride(
        self,
        backbone: tf.keras.Model,
        input_tensor: tf.Tensor,
        backbone_output: tf.Tensor,
    ) -> Dict[int, tf.Tensor]:
        """Expose stride-4/8/16/32 feature maps from the custom V4 backbone.

        The same shape-based helper used for V1/V2/V3 also works here:
        V4's UIB blocks wrap standard Keras layers internally, so the
        intermediate spatial resolutions are discoverable via layer
        output shapes.
        """
        return features_by_stride_from_keras_model(
            backbone,
            input_tensor,
            self.config.input_shape,
        )


def create_mobilenet_v4_qat_multi_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Create default configurations for multi-head MobileNetV4 QAT variants."""
    configs = []

    alpha_configs = [
        (0.25, "Ultra-lightweight MCU (~250KB)"),
        (0.50, "Lightweight MCU (~1MB)"),
        (0.75, "Balanced mobile (~2.25MB)"),
        (1.0, "Standard mobile (~4MB)"),
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
            arch_name = f"mobilenet_v4_qat_multi_{alpha_str}_{head_str}"

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
                'input_shape': (96, 96, 1),
                'num_classes': 2,
                'arch_params': {
                    'alpha': alpha,
                    'use_pretrained': False,  # V4 has no ImageNet weights here
                    'dropout_rate': 0.2,
                },
                'head_configs': head_configs_list,
                'training_mode': 'joint',
                'inference_mode': 'all_active',
                'description': (
                    f"MobileNetV4 QAT Multi-head with alpha={alpha}, "
                    f"heads={head_list} - {alpha_description}, {head_description}"
                ),
            }
            configs.append((arch_name, default_config))

    return configs


def _register_mobilenet_v4_qat_multi_variants():
    """Register all multi-head MobileNetV4 QAT variants with the factory."""
    for arch_name, default_config in create_mobilenet_v4_qat_multi_configs():
        ModelArchitectureFactory.register_architecture(
            name=arch_name,
            architecture_class=MultiHeadMobileNetV4QATArchitecture,
            default_config=default_config,
        )


_register_mobilenet_v4_qat_multi_variants()
