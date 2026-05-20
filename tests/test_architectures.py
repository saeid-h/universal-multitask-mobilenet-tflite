"""Integration tests for each multi-head architecture.

These build a real TF model (small input, small alpha) so they're slower
than the pure-unit tests in this directory. Each test verifies the
backbone-specific validate_config alpha rules, the _features_by_stride
output, and that build_model produces the expected number of outputs.
"""

import pytest
import tensorflow as tf

from models.components.head_configuration import HeadConfiguration
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.architectures.mobilenet_v1_qat_multi import MultiHeadMobileNetV1QATArchitecture
from models.architectures.mobilenet_v2_qat_multi import MultiHeadMobileNetV2QATArchitecture
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture
from models.architectures.mobilenet_v3_small_qat_multi import (
    MultiHeadMobileNetV3SmallCustomQATArchitecture,
)
from models.architectures.mobilenet_v4_qat_multi import MultiHeadMobileNetV4QATArchitecture


def _config(alpha, heads, input_shape=(96, 96, 3), use_pretrained=False):
    return MultiHeadModelConfig(
        input_shape=input_shape,
        head_configs=heads,
        arch_params={'alpha': alpha, 'use_pretrained': use_pretrained},
    )


@pytest.fixture
def two_heads():
    return [
        HeadConfiguration(name='h1', num_classes=5, activation='linear'),
        HeadConfiguration(name='h2', num_classes=2, activation='linear'),
    ]


# ----- Alpha-validation rejections (CPU-only, no model build) -----

def test_v1_rejects_alpha_035(two_heads):
    cfg = _config(0.35, two_heads)
    with pytest.raises(ValueError, match="MobileNetV1 supported values"):
        MultiHeadMobileNetV1QATArchitecture(cfg).validate_config()


def test_v2_accepts_alpha_035(two_heads):
    cfg = _config(0.35, two_heads)
    MultiHeadMobileNetV2QATArchitecture(cfg).validate_config()  # no raise


def test_v3_rejects_alpha_035(two_heads):
    cfg = _config(0.35, two_heads)
    with pytest.raises(ValueError, match="MobileNetV3"):
        MultiHeadMobileNetV3QATArchitecture(cfg).validate_config()


def test_v3c_rejects_alpha_035(two_heads):
    cfg = _config(0.35, two_heads, input_shape=(96, 96, 1))
    with pytest.raises(ValueError, match="custom V3-Small"):
        MultiHeadMobileNetV3SmallCustomQATArchitecture(cfg).validate_config()


def test_v4_rejects_alpha_035(two_heads):
    cfg = _config(0.35, two_heads)
    with pytest.raises(ValueError, match="MobileNetV4"):
        MultiHeadMobileNetV4QATArchitecture(cfg).validate_config()


# ----- _features_by_stride for each backbone (requires building backbone) -----

@pytest.mark.parametrize("cls,alpha,shape", [
    (MultiHeadMobileNetV1QATArchitecture, 0.25, (96, 96, 3)),
    (MultiHeadMobileNetV2QATArchitecture, 0.35, (96, 96, 3)),
    (MultiHeadMobileNetV3QATArchitecture, 0.25, (96, 96, 3)),
    (MultiHeadMobileNetV3SmallCustomQATArchitecture, 0.25, (96, 96, 1)),
    (MultiHeadMobileNetV4QATArchitecture, 0.25, (96, 96, 1)),
])
def test_features_by_stride_exposes_4_8_16_32(cls, alpha, shape):
    heads = [HeadConfiguration(name='h1', num_classes=2, activation='linear')]
    arch = cls(_config(alpha, heads, input_shape=shape))
    backbone = arch.build_backbone()
    input_tensor = tf.keras.layers.Input(shape=shape)
    backbone_output = backbone(input_tensor)
    features = arch._features_by_stride(backbone, input_tensor, backbone_output)
    # Every backbone should expose at least stride 32 and ideally 8/16/32.
    assert 32 in features, f"{cls.__name__} missing stride 32, got {sorted(features.keys())}"
    # Most backbones expose 8/16/32 too; let's confirm coverage.
    assert {8, 16, 32}.issubset(features.keys()), (
        f"{cls.__name__} should expose 8/16/32, got {sorted(features.keys())}"
    )


# ----- Mixed-stride routing via build_model -----

def test_v3_build_model_mixed_strides():
    heads = [
        HeadConfiguration(name='h32', num_classes=5, activation='linear', tap_stride=None),
        HeadConfiguration(name='h16', num_classes=2, activation='linear', tap_stride=16),
        HeadConfiguration(name='h8',  num_classes=3, activation='linear', tap_stride=8),
    ]
    arch = MultiHeadMobileNetV3QATArchitecture(_config(0.25, heads))
    model = arch.get_model()
    assert set(model.output.keys()) == {'h32', 'h16', 'h8'}
    # All standard heads end with GAP -> Dense, so each output is (None, num_classes).
    assert model.output['h32'].shape[-1] == 5
    assert model.output['h16'].shape[-1] == 2
    assert model.output['h8'].shape[-1] == 3


def test_invalid_tap_stride_rejected():
    heads = [HeadConfiguration(name='bad', num_classes=2, activation='linear', tap_stride=99)]
    arch = MultiHeadMobileNetV3QATArchitecture(_config(0.25, heads))
    with pytest.raises(ValueError, match="tap_stride=99"):
        arch.get_model()


def test_unified_output_rejects_spatial_taps():
    heads = [
        HeadConfiguration(name='h32', num_classes=5, activation='linear', tap_stride=None),
        HeadConfiguration(name='h8',  num_classes=3, activation='linear', tap_stride=8),
    ]
    arch = MultiHeadMobileNetV3QATArchitecture(_config(0.25, heads))
    arch._unified_output = True  # bypass single-head ignore path
    with pytest.raises(ValueError, match="unified_output=True is incompatible"):
        arch.get_model()
