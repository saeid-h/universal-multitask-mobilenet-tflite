"""Unit tests for src/utils/quantization.py representative-dataset helpers."""

import numpy as np
import pytest
from PIL import Image

from src.utils.quantization import (
    _list_images_in_dir,
    create_representative_dataset,
)


@pytest.fixture
def empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for i in range(4):
        arr = (np.random.RandomState(i).rand(64, 64, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(d / f"img_{i}.png")
    return d


def test_list_images_finds_pngs(image_dir):
    images = _list_images_in_dir(str(image_dir))
    assert len(images) == 4
    assert all(p.suffix == ".png" for p in images)


def test_list_images_rejects_missing(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="not a directory"):
        _list_images_in_dir(str(missing))


def test_list_images_rejects_empty(empty_dir):
    with pytest.raises(ValueError, match="contains no images"):
        _list_images_in_dir(str(empty_dir))


def test_list_images_ignores_non_images(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "not_an_image.txt").write_text("hello")
    arr = (np.random.RandomState(0).rand(8, 8, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(d / "valid.png")
    images = _list_images_in_dir(str(d))
    assert len(images) == 1
    assert images[0].suffix == ".png"


def test_random_generator_default(tmp_path):
    gen = create_representative_dataset((32, 32, 3), num_samples=3)
    batches = list(gen())
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 1
        assert batch[0].shape == (1, 32, 32, 3)
        assert batch[0].dtype == np.float32
        # Random data range is [0, 1].
        assert batch[0].min() >= 0.0
        assert batch[0].max() <= 1.0


def test_image_generator_loads_and_cycles(image_dir):
    # 4 images, ask for 10 samples -> should cycle.
    gen = create_representative_dataset(
        (32, 32, 3), num_samples=10, image_dir=str(image_dir),
    )
    batches = list(gen())
    assert len(batches) == 10
    for batch in batches:
        assert batch[0].shape == (1, 32, 32, 3)
        assert batch[0].dtype == np.float32
        # Image data normalized to [0, 1].
        assert batch[0].min() >= 0.0
        assert batch[0].max() <= 1.0


def test_image_generator_grayscale(image_dir):
    gen = create_representative_dataset(
        (32, 32, 1), num_samples=2, image_dir=str(image_dir),
    )
    batches = list(gen())
    assert len(batches) == 2
    for batch in batches:
        assert batch[0].shape == (1, 32, 32, 1)
