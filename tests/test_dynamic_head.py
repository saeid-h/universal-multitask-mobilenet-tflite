"""Integration test for add_head_dynamically with tap_stride + frozen backbone."""

import pytest

from models.components.head_configuration import HeadConfiguration
from models.components.multi_head_model_config import MultiHeadModelConfig
from models.architectures.mobilenet_v3_qat_multi import MultiHeadMobileNetV3QATArchitecture


def test_add_head_with_different_stride_and_frozen_backbone():
    initial = [
        HeadConfiguration(name=f'h{i}', num_classes=c, activation='linear')
        for i, c in enumerate([5, 2, 3])
    ]
    arch = MultiHeadMobileNetV3QATArchitecture(MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=initial,
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        separable_weights=True,
    ))
    arch.get_model()
    assert arch.head_names == ['h0', 'h1', 'h2']

    new_model = arch.add_head_dynamically(
        num_classes=7,
        head_name='h_new_stride16',
        tap_stride=16,
        freeze_backbone=True,
    )
    # New head registered.
    assert arch.head_names == ['h0', 'h1', 'h2', 'h_new_stride16']
    # Backbone really is frozen.
    assert arch.backbone.trainable is False
    # All four outputs are present in the new model.
    assert set(new_model.output.keys()) == {'h0', 'h1', 'h2', 'h_new_stride16'}
    # New head produces (None, 7) since the standard builder GAPs.
    assert new_model.output['h_new_stride16'].shape[-1] == 7


def test_add_head_rejects_duplicate_name():
    initial = [HeadConfiguration(name='only', num_classes=2, activation='linear')]
    arch = MultiHeadMobileNetV3QATArchitecture(MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=initial,
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        separable_weights=True,
    ))
    arch.get_model()
    with pytest.raises(ValueError, match="already exists"):
        arch.add_head_dynamically(num_classes=3, head_name='only')


def test_add_head_requires_separable_weights():
    initial = [HeadConfiguration(name='only', num_classes=2, activation='linear')]
    arch = MultiHeadMobileNetV3QATArchitecture(MultiHeadModelConfig(
        input_shape=(96, 96, 3),
        head_configs=initial,
        arch_params={'alpha': 0.25, 'use_pretrained': False},
        separable_weights=False,
    ))
    arch.get_model()
    with pytest.raises(ValueError, match="requires separable_weights"):
        arch.add_head_dynamically(num_classes=3, head_name='other')
