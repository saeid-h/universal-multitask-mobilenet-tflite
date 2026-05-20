"""Unit tests for CLI argument parsing helpers in create_quantized_mobilenet.py."""

import pytest

from src.create_quantized_mobilenet import _parse_head_config


def test_simple_heads_string():
    classes, names, strides = _parse_head_config("5,2,3")
    assert classes == [5, 2, 3]
    assert names is None
    assert strides == [None, None, None]


def test_heads_with_at_stride():
    classes, names, strides = _parse_head_config("5@32,2@16,3@8")
    assert classes == [5, 2, 3]
    assert strides == [32, 16, 8]


def test_heads_mixed_at_and_bare():
    classes, names, strides = _parse_head_config("5,2@16,3")
    assert classes == [5, 2, 3]
    assert strides == [None, 16, None]


def test_heads_with_explicit_names():
    classes, names, strides = _parse_head_config("5,2", head_names_str="cls,det")
    assert classes == [5, 2]
    assert names == ['cls', 'det']


def test_heads_name_count_mismatch_raises():
    with pytest.raises(ValueError, match="Number of head names"):
        _parse_head_config("5,2,3", head_names_str="only_two,names")


def test_negative_class_count_rejected():
    with pytest.raises(ValueError, match="Invalid head configuration"):
        _parse_head_config("5,-2,3")


def test_zero_class_count_rejected():
    with pytest.raises(ValueError, match="Invalid head configuration"):
        _parse_head_config("5,0,3")


def test_zero_stride_rejected():
    with pytest.raises(ValueError, match="tap_stride must be a positive integer"):
        _parse_head_config("5@0")


def test_negative_stride_rejected():
    with pytest.raises(ValueError, match="tap_stride must be a positive integer"):
        _parse_head_config("5@-8")


def test_non_int_class_rejected():
    with pytest.raises(ValueError, match="Invalid head configuration"):
        _parse_head_config("five,2")


def test_non_int_stride_rejected():
    with pytest.raises(ValueError, match="Invalid head configuration"):
        _parse_head_config("5@thirty-two")


def test_whitespace_tolerated():
    classes, _, strides = _parse_head_config(" 5@32 , 2@16 , 3 ")
    assert classes == [5, 2, 3]
    assert strides == [32, 16, None]


def test_empty_string_rejected():
    with pytest.raises(ValueError):
        _parse_head_config("")
