import pytest
import torch

import bwb.distributions.utils as utils

_log = _logging.get_logger(__name__)

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

dtypes = [torch.float16, torch.float32, torch.float64]


@pytest.fixture(params=devices)
def device(request):
    """Fixture to test the different devices."""
    return request.param


@pytest.fixture(params=dtypes)
def dtype(request):
    """Fixture to test the different data types."""
    return request.param


@pytest.fixture
def gen(device, dtype) -> tuple[torch.Generator, str, torch.dtype]:
    """
    Fixture to generate a generator with a seed and device.

    :param device: The device of the generator.
    :param dtype: The data type of the generator.
    :return: The generator with the device and data type.
    """
    _logging.set_level(_logging.DEBUG)
    seed = torch.Generator().seed()
    _log.info(f"Seed: {seed}, device: {device}, dtype: {dtype}")
    return torch.Generator(device=device).manual_seed(seed), device, dtype


def test_grayscale_parser_invalid_shape(gen):
    gen, device, _ = gen
    shape = (30, 30)
    weights = torch.rand((784,), device=device, generator=gen)
    support = torch.randint(0, 28, (784, 2), device=device, generator=gen)
    with pytest.raises(ValueError):
        utils.grayscale_parser(shape, weights, support)


def test_partition_invalid_alpha(gen):
    gen, device, _ = gen
    X = torch.rand((784, 2), device=device, generator=gen)
    mu = torch.rand((784,), device=device, generator=gen)
    alpha = -0.5
    with pytest.raises(ValueError):
        utils.partition(X, mu, alpha)


# noinspection PyTestUnpassedFixture
def test_grayscale_parser_valid_input(gen):
    generator, device, dtype = gen
    shape = (28, 28)
    weights = torch.rand(
        (784,), dtype=dtype, device=device, generator=generator
    )
    support = torch.randint(
        0, 28, (784, 2),
        dtype=torch.int32, device=device, generator=generator
    )
    result = utils.grayscale_parser(shape, weights, support)
    assert result.shape == shape
    assert result.dtype == torch.uint8
    assert result.device.type == device


# noinspection PyTestUnpassedFixture
def test_partition_valid_input(gen):
    generator, device, dtype = gen
    X = torch.rand(
        (784, 2), dtype=dtype, device=device, generator=generator
    )
    mu = torch.rand(
        (784,), dtype=dtype, device=device, generator=generator
    )
    alpha = 0.1

    if dtype == torch.float16:
        with pytest.raises(ValueError):
            utils.partition(X, mu, alpha)
        return None

    X_new, mu_new = utils.partition(X, mu, alpha)
    assert X_new.shape[0] >= X.shape[0]
    assert mu_new.shape[0] >= mu.shape[0]
    assert torch.isclose(
        torch.sum(mu_new),
        torch.tensor(1.0, device=device, dtype=dtype)
    )
