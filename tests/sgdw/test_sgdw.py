import typing as t

import pytest
import torch

import bwb.distributions as D
import bwb.logging_ as logging
from bwb.sgdw import sgdw, utils

_log = logging.get_logger(__name__)
logging.set_level(logging.DEBUG, name=__name__)

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices.append(torch.device('cuda:0'))
else:
    logging.raise_warning(
        "CUDA is not available. Skipping tests for CUDA devices.",
        _log, stacklevel=2
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
    logging.set_level(logging.DEBUG)
    seed = torch.Generator().seed()
    _log.info(f"Seed: {seed}, device: {device}, dtype: {dtype}")
    return torch.Generator(device=device).manual_seed(seed), device, dtype


# noinspection PyMissingOrEmptyDocstring
class MockDistributionSampler[DistributionT]:
    """
    Mock class for the distribution sampler.
    """

    def __init__(
        self,
        distributions: list[DistributionT],
        gen: torch.Generator,
    ):
        self.distributions = distributions
        self.gen = gen

    def sample(self, n: int) -> t.Sequence[DistributionT]:
        # Generate a list of n elements selecting randomly n elements
        # (with replacement) from the distributions list.
        index = torch.randint(
            0, len(self.distributions), (n,), generator=self.gen,
            device=self.device, dtype=torch.int64
        )
        return [self.distributions[int(i)] for i in index]

    def draw(self) -> DistributionT:
        # Select randomly a distribution from the distributions list.
        index = torch.randint(
            0, len(self.distributions), (1,), generator=self.gen,
            device=self.device, dtype=torch.int64
        )
        return self.distributions[int(index)]

    @property
    def device(self) -> torch.device:
        return self.distributions[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.distributions[0].dtype


# noinspection PyMissingTypeHints
@pytest.fixture
def distribution_sampler(gen):
    """
    Fixture to generate a distribution sampler.
    """
    gen, device, dtype = gen
    distributions: [D.DistributionDraw] = [
        D.DistributionDraw.from_array(
            torch.randint(0, 256, (28, 28),
                          dtype=torch.uint8, device=device, generator=gen),
            dtype=dtype
        ) for _ in range(10)
    ]
    return MockDistributionSampler(distributions, gen)


def test_sgdw_initialization(distribution_sampler) -> None:
    sgdw_instance = sgdw.DistributionDrawSGDW(
        distribution_sampler,
        utils.step_scheduler(),
        conv_bar_strategy="conv",
    )
    assert sgdw_instance is not None
    assert sgdw_instance.iter_params.k == 0
    assert sgdw_instance.distr_sampler == distribution_sampler
    assert sgdw_instance.dtype == distribution_sampler.dtype
    assert sgdw_instance.device == distribution_sampler.device


def test_sgdw_init_algorithm(distribution_sampler) -> None:
    sgdw_instance = sgdw.DistributionDrawSGDW(
        distribution_sampler,
        utils.step_scheduler(),
        batch_size=5,
        conv_bar_strategy="conv",
    )
    lst_mu_0, pos_wgt_0 = sgdw_instance.init_algorithm()

    assert isinstance(lst_mu_0, list)
    assert isinstance(lst_mu_0[0], D.DistributionDraw)
    assert isinstance(pos_wgt_0, torch.Tensor)

    assert len(lst_mu_0) == 1
    assert pos_wgt_0.shape == (28, 28)
    assert pos_wgt_0.dtype == distribution_sampler.dtype
    assert pos_wgt_0.device == distribution_sampler.device


def test_sgdw_step_algorithm(distribution_sampler) -> None:
    sgdw_instance = sgdw.DistributionDrawSGDW(
        distribution_sampler,
        utils.step_scheduler(),
        batch_size=5,
        conv_bar_strategy="conv",
    )
    lst_mu_0, pos_wgt_0 = sgdw_instance.init_algorithm()
    lst_mu_1, pos_wgt_1 = sgdw_instance.step_algorithm(1, pos_wgt_0)

    assert isinstance(lst_mu_1, list)
    assert isinstance(lst_mu_1[0], D.DistributionDraw)
    assert isinstance(pos_wgt_1, torch.Tensor)

    assert len(lst_mu_1) == 5
    assert pos_wgt_1.shape == (28, 28)
    assert pos_wgt_1.dtype == distribution_sampler.dtype
    assert pos_wgt_1.device == distribution_sampler.device


def test_sgdw_run(distribution_sampler) -> None:
    sgdw_instance = sgdw.DistributionDrawSGDW(
        distribution_sampler,
        utils.step_scheduler(b=1),
        batch_size=5,
        conv_bar_strategy="conv",
        max_iter=10,
    )
    result = sgdw_instance.run()

    assert isinstance(result, D.DistributionDraw)
    assert result.shape == (28, 28)
    assert result.dtype == distribution_sampler.dtype
    assert result.device == distribution_sampler.device
    assert sgdw_instance.iter_params.k == 10
