"""Unit tests for HeadConfiguration and its tap_stride field."""

import pytest

from models.components.head_configuration import (
    HeadConfiguration,
    create_classification_head,
    create_segmentation_head,
    create_head_config_from_list,
)


def test_default_construction():
    h = HeadConfiguration(name='cls', num_classes=5)
    assert h.name == 'cls'
    assert h.num_classes == 5
    assert h.activation == 'softmax'
    assert h.dropout_rate == 0.2
    assert h.head_type == 'standard'
    assert h.tap_stride is None  # default = use final feature map


def test_tap_stride_accepts_int():
    h = HeadConfiguration(name='seg', num_classes=10, head_type='segmentation', tap_stride=8)
    assert h.tap_stride == 8


def test_tap_stride_none_is_default():
    h = HeadConfiguration(name='cls', num_classes=2)
    assert h.tap_stride is None


def test_tap_stride_rejects_zero():
    with pytest.raises(ValueError, match="tap_stride must be a positive integer"):
        HeadConfiguration(name='bad', num_classes=2, tap_stride=0)


def test_tap_stride_rejects_negative():
    with pytest.raises(ValueError, match="tap_stride must be a positive integer"):
        HeadConfiguration(name='bad', num_classes=2, tap_stride=-4)


def test_tap_stride_rejects_non_int():
    with pytest.raises(ValueError, match="tap_stride must be a positive integer"):
        HeadConfiguration(name='bad', num_classes=2, tap_stride="32")  # type: ignore[arg-type]


def test_invalid_num_classes():
    with pytest.raises(ValueError, match="num_classes must be positive"):
        HeadConfiguration(name='cls', num_classes=0)


def test_invalid_activation():
    with pytest.raises(ValueError, match="activation must be one of"):
        HeadConfiguration(name='cls', num_classes=2, activation='bogus')


def test_invalid_head_type():
    with pytest.raises(ValueError, match="head_type must be one of"):
        HeadConfiguration(name='cls', num_classes=2, head_type='not_a_head_type')


def test_to_dict_includes_tap_stride():
    h = HeadConfiguration(name='seg', num_classes=10, head_type='segmentation', tap_stride=16)
    d = h.to_dict()
    assert d['tap_stride'] == 16
    assert d['head_type'] == 'segmentation'


def test_create_classification_head_accepts_tap_stride():
    h = create_classification_head('cls', num_classes=5, tap_stride=32)
    assert h.tap_stride == 32
    assert h.head_type == 'standard'


def test_create_segmentation_head_accepts_tap_stride():
    h = create_segmentation_head('seg', num_classes=7, tap_stride=8)
    assert h.tap_stride == 8
    assert h.head_type == 'segmentation'


def test_create_head_config_from_list_basic():
    heads = create_head_config_from_list([5, 2, 3])
    assert len(heads) == 3
    assert heads[0].num_classes == 5
    assert heads[2].num_classes == 3
    assert all(h.head_type == 'standard' for h in heads)
    assert all(h.tap_stride is None for h in heads)


def test_create_head_config_from_list_with_names():
    heads = create_head_config_from_list([5, 2], head_names=['a', 'b'])
    assert [h.name for h in heads] == ['a', 'b']


def test_create_head_config_from_list_name_count_mismatch():
    with pytest.raises(ValueError, match="head_names length"):
        create_head_config_from_list([5, 2, 3], head_names=['only_one'])
