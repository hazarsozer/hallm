import numpy as np
import pytest
import torch

from hallm.data.imagenet import IMAGENET_MEAN, get_image_batch, load_images, load_labels, preprocess_image


def _fixture(tmp_path, n=8):
    images = np.random.default_rng(0).integers(0, 255, (n, 3, 128, 128), dtype=np.uint8)
    labels = np.arange(n, dtype=np.int64)
    images.tofile(tmp_path / "train.bin")
    np.save(tmp_path / "train_labels.npy", labels)
    return tmp_path


def test_preprocess_resizes_shorter_side_and_center_crops():
    Image = pytest.importorskip("PIL.Image")
    img = Image.new("RGB", (320, 160), color=(10, 20, 30))
    out = preprocess_image(img, size=128)
    assert out.shape == (3, 128, 128)
    assert out.dtype == np.uint8
    assert int(out[0].mean()) == 10


def test_preprocess_handles_grayscale_input():
    Image = pytest.importorskip("PIL.Image")
    out = preprocess_image(Image.new("L", (200, 200), color=128), size=128)
    assert out.shape == (3, 128, 128)


def test_load_images_reads_the_memmap_shape(tmp_path):
    d = _fixture(tmp_path)
    images = load_images(d / "train.bin")
    assert images.shape == (8, 3, 128, 128) and images.dtype == np.uint8
    assert load_labels(d / "train_labels.npy").shape == (8,)


def test_batch_shapes_and_normalization(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    gen = torch.Generator().manual_seed(0)
    x, y = get_image_batch(images, labels, batch_size=4, crop=112, device="cpu", generator=gen)
    assert x.shape == (4, 3, 112, 112) and x.dtype == torch.float32
    assert y.shape == (4,) and y.dtype == torch.int64
    assert abs(float(x.mean())) < 3.0        # normalized, not raw 0-255
    assert float(x.max()) < 10.0


def test_batches_are_reproducible_given_a_seeded_generator(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    a = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(7))
    b = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_eval_batches_are_deterministic_center_crops(tmp_path):
    d = _fixture(tmp_path)
    images, labels = load_images(d / "train.bin"), load_labels(d / "train_labels.npy")
    a = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(1), train=False)
    b = get_image_batch(images, labels, 4, 112, "cpu", torch.Generator().manual_seed(2), train=False)
    assert torch.equal(a[0], b[0])           # no crop jitter, no flip, no shuffling
    assert torch.equal(a[1], torch.arange(4))


def test_mean_constant_is_the_standard_imagenet_one():
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
