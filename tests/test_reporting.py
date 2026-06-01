"""Unit tests for src/utils/reporting.py helpers (no TF model required)."""

import pytest

from src.utils.reporting import generate_output_name


def test_default_backbone_v3_prefix():
    name = generate_output_name(0.25, (96, 96, 3), [5, 2, 3])
    assert name == 'mnv3_0_25_5_2_3_96x96x3'


@pytest.mark.parametrize("backbone,prefix", [
    ('v1', 'mnv1'),
    ('v2', 'mnv2'),
    ('v3', 'mnv3'),
    ('v3c', 'mnv3c'),
    ('v4', 'mnv4'),
])
def test_backbone_prefix_propagates(backbone, prefix):
    name = generate_output_name(0.25, (96, 96, 3), [5, 2, 3], backbone=backbone)
    assert name.startswith(prefix + '_')


def test_alpha_decimal_replaced_with_underscore():
    name = generate_output_name(0.35, (128, 128, 3), [2], backbone='v2')
    assert '0_35' in name
    assert '.' not in name


def test_input_shape_embedded():
    name = generate_output_name(1.0, (224, 224, 1), [10], backbone='v1')
    assert '224x224x1' in name


def test_heads_joined_with_underscores():
    name = generate_output_name(0.25, (96, 96, 1), [5, 2, 3, 4], backbone='v3')
    assert '_5_2_3_4_' in name
