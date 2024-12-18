"""
Tests for the :mod:`bwb.distributions.discrete_distribution` module.
"""

import pytest
import torch

import bwb.distributions.discrete_distribution as dd
import bwb.logging_ as logging

_log = logging.get_logger(__name__)
logging.set_level(logging.DEBUG, name=__name__)

devices = [torch.device("cpu")]
if torch.cuda.is_available():
    devices.append(torch.device("cuda:0"))
else:
    logging.raise_warning(
        "CUDA is not available. Skipping tests for CUDA devices.",
        _log,
        stacklevel=2,
    )

dtypes = [torch.float32, torch.float64]


@pytest.fixture(params=devices)
def device(request) -> torch.device:
    """Fixture to test the different devices."""
    return request.param


@pytest.fixture(params=dtypes)
def dtype(request) -> torch.dtype:
    """Fixture to test the different data types."""
    return request.param


@pytest.fixture
def gen(device, dtype) -> tuple[torch.Generator, torch.device, torch.dtype]:
    """
    Fixture to generate a generator with a seed and device.

    :param device: The device of the generator.
    :param dtype: The data type of the generator.
    :return: The generator with the device and data type.
    """
    seed = torch.Generator().seed()
    _log.info(f"Seed: {seed}, device: {device}, dtype: {dtype}")
    return torch.Generator(device=device).manual_seed(seed), device, dtype


def test_discrete_distribution_initialization(gen) -> None:
    gen, device, dtype = gen
    weights = torch.rand((10,), device=device, dtype=dtype, generator=gen)
    support = torch.randint(0, 28, (10,), device=device, generator=gen)
    dist = dd.DiscreteDistribution(weights, support)
    assert torch.all(dist.weights == weights)
    assert torch.all(dist.original_support == support)


def test_discrete_distribution_without_support(gen) -> None:
    gen, device, dtype = gen
    weights = torch.rand((10,), device=device, dtype=dtype, generator=gen)
    dist = dd.DiscreteDistribution(weights)
    assert torch.all(dist.weights == weights)
    assert torch.all(
        dist.original_support == torch.arange(len(weights), device=device)
    )


def test_discrete_distribution_invalid_weights_support(gen) -> None:
    gen, device, dtype = gen
    weights = torch.tensor([0.1, 0.2, 0.7], device=device, dtype=dtype)
    support = torch.tensor([1, 2], device=device, dtype=dtype)
    with pytest.raises(ValueError):
        dd.DiscreteDistribution(weights, support)


def test_discrete_distribution_properties(gen) -> None:
    gen, device, dtype = gen
    weights = torch.rand((10,), device=device, dtype=dtype, generator=gen)
    weights[0] = 0
    support = torch.randint(0, 28, (10,), device=device, generator=gen)
    dist = dd.DiscreteDistribution(weights, support)
    assert dist.device == device
    assert dist.dtype == dtype


def test_distribution_draw_initialization_with_shape(gen) -> None:
    gen, device, dtype = gen
    shape = (2, 1)
    weights = torch.rand(shape, device=device, dtype=dtype, generator=gen)
    dist = dd.DistributionDraw(weights, shape)
    assert dist.shape == shape
    assert torch.equal(dist.weights, weights)


# noinspection PyDeprecation
def test_distribution_draw_from_weights(gen) -> None:
    gen, device, dtype = gen
    shape = (10, 10)
    weights = torch.rand(shape, device=device, dtype=dtype, generator=gen)
    dist = dd.DistributionDraw.from_weights(weights, shape)
    assert torch.all(dist.weights == weights)
    assert dist.shape == shape
    shape = (2, 2)
    weights = torch.rand(shape, device=device, dtype=dtype, generator=gen)
    original_support = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], device=device
    )
    dist = dd.DistributionDraw.from_weights(weights, shape)

    assert torch.all(dist.original_support == original_support)


def test_distribution_draw_from_discrete_distribution(gen) -> None:
    gen, device, dtype = gen
    shape = (10, 10)
    weights = torch.rand(shape, device=device, dtype=dtype, generator=gen)
    support = torch.randint(0, 28, shape, device=device, generator=gen)
    discrete_dist = dd.DiscreteDistribution(weights, support)
    dist = dd.DistributionDraw.from_discrete_distribution(discrete_dist, shape)
    assert torch.all(dist.weights == weights)
    assert torch.all(dist.original_support == support)
    assert dist.shape == shape


def test_distribution_draw_from_grayscale_weights(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [
            [0, 128, 255],
            [64, 192, 128],
            [128, 64, 192],
        ],
        dtype=torch.uint8,
        device=device,
    )
    grayscale_weights = grayscale.to(dtype) / 255
    grayscale_weights /= torch.sum(grayscale_weights)
    dist = dd.DistributionDraw.from_grayscale_weights(grayscale_weights)
    assert dist.shape == grayscale.shape
    assert torch.allclose(dist.grayscale, grayscale, atol=1e-5)
    assert torch.allclose(dist.grayscale_weights, grayscale_weights, atol=1e-5)


def test_distribution_draw_from_array(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [
            [0, 128, 255],
            [64, 192, 128],
            [128, 64, 192],
        ],
        dtype=torch.uint8,
        device=device,
    )
    dist = dd.DistributionDraw.from_array(grayscale, dtype=dtype)

    assert dist.shape == grayscale.shape
    assert torch.allclose(dist.grayscale, grayscale, atol=1e-5)

    grayscale_weights = grayscale.to(dtype) / 255
    grayscale_weights /= torch.sum(grayscale_weights)
    assert torch.allclose(dist.grayscale_weights, grayscale_weights, atol=1e-5)


def test_distribution_draw_from_array_creates_correct_weights(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [[0, 128, 255], [64, 192, 128], [128, 64, 192]],
        dtype=torch.uint8,
        device=device,
    )
    dist = dd.DistributionDraw.from_array(grayscale, dtype=dtype)
    expected_weights = torch.tensor(
        [0, 128, 255, 64, 192, 128, 128, 64, 192], dtype=dtype, device=device
    ) / torch.sum(grayscale.to(dtype))
    assert torch.allclose(dist.weights, expected_weights, atol=1e-5)

    assert dist.shape == grayscale.shape
    assert torch.allclose(dist.grayscale, grayscale, atol=1e-5)

    grayscale_weights = grayscale.to(dtype) / 255
    grayscale_weights /= torch.sum(grayscale_weights)
    assert torch.allclose(dist.grayscale_weights, grayscale_weights, atol=1e-5)


def test_distribution_draw_image_returns_pil_image(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [
            [0, 128, 255],
            [64, 192, 128],
            [128, 64, 192],
            [255, 255, 255],
        ],
        dtype=torch.uint8,
        device=device,
    )
    dist = dd.DistributionDraw.from_array(grayscale, dtype=dtype)
    image = dist.image
    assert image.size == grayscale.shape[::-1]
    assert image.mode == "L"


def test_grayscale_property_returns_existing_grayscale(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [
            [0, 128, 255],
            [64, 192, 128],
            [128, 64, 192],
            [255, 255, 255],
        ],
        dtype=torch.uint8,
        device=device,
    )

    # Directly setting the cached value
    dist = dd.DistributionDraw(
        weights=torch.tensor(
            torch.rand(grayscale.shape, device=device, dtype=dtype)
        ),
        shape=grayscale.shape,
    )
    dist._grayscale = grayscale
    assert torch.allclose(dist.grayscale, grayscale, atol=1e-5)


def test_grayscale_property_computes_grayscale_from_weights(gen) -> None:
    gen, device, dtype = gen
    weights = torch.tensor([1.0, 0.0, 0.5, 0.5], device=device, dtype=dtype)
    support = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)
    shape = (2, 2)
    dist = dd.DistributionDraw(weights, shape, support)
    expected_grayscale = torch.tensor(
        [[255, 0], [127, 127]], dtype=torch.uint8, device=device
    )
    assert torch.equal(dist.grayscale, expected_grayscale)


def test_grayscale_property_normalizes_grayscale_weights_correctly(gen) -> None:
    gen, device, dtype = gen
    grayscale_weights = torch.tensor(
        [[0.2, 0.8], [0.1, 1.0]], dtype=dtype, device=device
    )
    grayscale_weights /= torch.sum(grayscale_weights)
    expected_grayscale = torch.tensor(
        [[51, 204], [25, 255]], dtype=torch.uint8, device=device
    )
    false_weights = torch.rand(
        (2, 2), generator=gen, device=device, dtype=dtype
    )
    dist = dd.DistributionDraw(weights=false_weights, shape=(2, 2))
    dist._grayscale_weights = grayscale_weights
    assert torch.equal(dist.grayscale, expected_grayscale)


def test_grayscale_property_handles_zero_max_grayscale_weight(gen) -> None:
    gen, device, dtype = gen
    grayscale_weights = torch.tensor(
        [[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device
    )
    expected_grayscale = torch.tensor(
        [[0, 255], [0, 0]], dtype=torch.uint8, device=device
    )
    false_weights = torch.rand(
        (2, 2), generator=gen, device=device, dtype=dtype
    )
    dist = dd.DistributionDraw(weights=false_weights, shape=(2, 2))
    dist._grayscale_weights = grayscale_weights
    assert torch.equal(dist.grayscale, expected_grayscale)


def test_grayscale_weights_property_returns_normalized_weights(gen) -> None:
    gen, device, dtype = gen
    grayscale = torch.tensor(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=torch.uint8,
        device=device,
    )
    expected_weights = (
        torch.tensor([10, 20, 30, 40], dtype=dtype, device=device) / 100
    )

    # dist = DistributionDraw.from_array(grayscale, dtype=dtype)
    dist = dd.DistributionDraw(
        weights=torch.rand((2, 2), device=device, dtype=dtype), shape=(2, 2)
    )
    dist._grayscale = grayscale

    assert torch.allclose(
        dist.grayscale_weights, expected_weights.view(2, 2), atol=1e-5
    )


def test_representation_includes_class_name_shape_device_dtype(gen) -> None:
    gen, device, dtype = gen
    weights = torch.rand((10, 10), device=device, dtype=dtype, generator=gen)
    dist = dd.DistributionDraw(weights, (10, 10))
    representation = repr(dist)
    assert "DistributionDraw" in representation
    assert "shape=(10, 10)" in representation
    assert f"device={device.type}" in representation
    assert f"dtype={dtype}" in representation


def test_representation_handles_empty_shape_correctly(gen) -> None:
    gen, device, dtype = gen
    dist = dd.DistributionDraw(
        torch.rand((0, 0)), (0, 0), dtype=dtype, device=device
    )
    representation = repr(dist)
    assert "DistributionDraw" in representation
    assert "shape=(0, 0)" in representation
    assert f"device={device.type}" in representation
    assert f"dtype={dtype}" in representation
