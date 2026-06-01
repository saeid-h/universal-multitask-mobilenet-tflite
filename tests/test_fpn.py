"""Tests for the opt-in FPN fusion mode."""

import pytest
import tensorflow as tf

from models.components.head_configuration import HeadConfiguration
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture


def _heads(strides):
    return [
        HeadConfiguration(
            name=f'h{i}', num_classes=2 + i, activation='linear',
            tap_stride=strides[i],
        )
        for i in range(len(strides))
    ]


def _build(heads, fusion=None, fpn_channels=128):
    cfg = MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=heads,
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        fusion=fusion,
        fpn_channels=fpn_channels,
    )
    arch = MultiHeadMobileNetV3QATArchitecture(cfg)
    return arch, arch.get_model()


def test_fpn_none_is_default():
    arch, _ = _build(_heads([None, None, None]))
    assert arch._fusion is None


def test_fpn_builds_lateral_at_every_used_stride():
    _, model = _build(_heads([None, 16, 8]), fusion='fpn')
    layer_names = {l.name for l in model.layers}
    # Three strides used (None -> coarsest = 32, plus 16, 8).
    assert 'fpn_lateral_32' in layer_names
    assert 'fpn_lateral_16' in layer_names
    assert 'fpn_lateral_8' in layer_names


def test_fpn_top_down_pathway_exists():
    _, model = _build(_heads([None, 16, 8]), fusion='fpn')
    layer_names = {l.name for l in model.layers}
    # Upsample + merge + smooth between consecutive levels.
    assert 'fpn_up_32_to_16' in layer_names
    assert 'fpn_merge_16' in layer_names
    assert 'fpn_smooth_16' in layer_names
    assert 'fpn_up_16_to_8' in layer_names
    assert 'fpn_merge_8' in layer_names
    assert 'fpn_smooth_8' in layer_names


def test_fpn_only_builds_used_strides():
    # Only heads at 32 and 8 -- no lateral at 16 should be built.
    _, model = _build(_heads([None, 8]), fusion='fpn')
    layer_names = {l.name for l in model.layers}
    assert 'fpn_lateral_32' in layer_names
    assert 'fpn_lateral_8' in layer_names
    assert 'fpn_lateral_16' not in layer_names


def test_fpn_channel_count_propagates():
    _, model = _build(_heads([None, 16]), fusion='fpn', fpn_channels=37)
    lateral_16 = model.get_layer('fpn_lateral_16')
    # Conv2D filters = fpn_channels
    assert lateral_16.filters == 37


def test_fpn_output_shapes_match_single_tap():
    # FPN should not change head output shapes (still 1D per head).
    _, fpn_model = _build(_heads([None, 16, 8]), fusion='fpn')
    _, plain_model = _build(_heads([None, 16, 8]), fusion=None)
    for key in fpn_model.output:
        assert fpn_model.output[key].shape[-1] == plain_model.output[key].shape[-1]


def test_fpn_config_rejects_unknown_mode():
    cfg = MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=_heads([None]),
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        fusion='bogus',
    )
    with pytest.raises(ValueError, match="fusion must be one of"):
        cfg.validate()


def test_fpn_config_rejects_zero_channels():
    cfg = MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=_heads([None]),
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        fusion='fpn',
        fpn_channels=0,
    )
    with pytest.raises(ValueError, match="fpn_channels must be"):
        cfg.validate()
