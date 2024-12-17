from math import prod
from pathlib import Path

import torch

import bwb.logging_ as logging
from bwb.config import config

__all__ = [
    "grayscale_parser",
    "partition",
]

_log = logging.get_logger(__name__)

_COMPILED_FUNC_PATH = Path(__file__).parent / "_compiled_funcs"
_COMPILED_FUNC_PATH.mkdir(exist_ok=True, parents=True)
_GRAYSCALE_PATH = _COMPILED_FUNC_PATH / "grayscale.pt"
_PARTITION_PATH = _COMPILED_FUNC_PATH / "partition.pt"


def __grayscale(
    to_return: torch.Tensor,
    weights: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    support1, support2 = support[:, 0], support[:, 1]
    for w, pos1, pos2 in zip(weights, support1, support2):
        to_return[pos1, pos2] += w
    to_return /= torch.max(to_return)
    to_return *= 255
    return to_return


if _GRAYSCALE_PATH.exists():
    _grayscale = torch.jit.load(_GRAYSCALE_PATH)
else:
    # noinspection PyUnreachableCode
    _grayscale = torch.jit.script(
        __grayscale,
        example_inputs=[
            (
                # to_return
                torch.rand((28, 28), dtype=torch.float32, device=config.device),
                # weights
                torch.rand((784,), dtype=torch.float32, device=config.device),
                # support
                torch.randint(
                    0,
                    28,
                    size=(784, 2),
                    dtype=torch.int32,
                    device=config.device,
                ),
            ),
            (
                # to_return
                torch.rand((28, 28), dtype=torch.float64, device=config.device),
                # weights
                torch.rand((784,), dtype=torch.float64, device=config.device),
                # support
                torch.randint(
                    0,
                    28,
                    size=(784, 2),
                    dtype=torch.int32,
                    device=config.device,
                ),
            ),
        ],
    )
    _grayscale.save(str(_GRAYSCALE_PATH), {})


def grayscale_parser(
    shape: tuple[int, ...],
    weights: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """
    Function that parses the weights and support into a grayscale image.

    :param shape: The shape of the image.
    :param weights: The weights of the distribution.
    :param support: The support of the distribution.
    :return: A grayscale image.
    """
    if prod(shape) != support.shape[0]:
        logging.raise_error(
            "The shape of the image must be equal to the number of samples",
            _log,
            ValueError,
        )
    device = weights.device
    dtype = weights.dtype
    support = torch.round(support).to(dtype=torch.int32, device=device)
    to_return = torch.zeros(shape, dtype=dtype, device=device)
    to_return: torch.Tensor = _grayscale(to_return, weights, support)

    return to_return.type(torch.uint8)


def __partition(
    X: torch.Tensor,
    mu: torch.Tensor,
    alpha,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, n_dim = X.shape
    min_w = torch.min(mu)

    n_times = torch.ceil(alpha * mu / min_w).type(torch.int).to(mu.device)
    n_rows = int(torch.sum(n_times))

    X_ = torch.zeros((n_rows, n_dim), dtype=X.dtype, device=X.device)
    mu_ = torch.zeros((n_rows,), dtype=mu.dtype, device=mu.device)
    i = 0
    for x, w, n in zip(X, mu, n_times):
        x, w = x, w / n
        for _ in range(int(n)):
            X_[i] = x
            mu_[i] = w
            i += 1

    return X_, mu_


if _PARTITION_PATH.exists():
    _partition = torch.jit.load(_PARTITION_PATH)
else:
    # noinspection PyUnreachableCode
    _partition = torch.jit.script(
        __partition,
        example_inputs=[
            (
                # X
                torch.rand((784, 2), dtype=torch.float32, device=config.device),
                # mu
                torch.rand((784,), dtype=torch.float32, device=config.device),
                0.5,  # alpha
            ),
            (
                # X
                torch.rand((784, 2), dtype=torch.float64, device=config.device),
                # mu
                torch.rand((784,), dtype=torch.float64, device=config.device),
                0.5,  # alpha
            ),
        ],
    )
    _partition.save(str(_PARTITION_PATH), {})


def partition(X: torch.Tensor, mu: torch.Tensor, alpha: float):
    """
    Function that partitions the samples in X according to the weights in mu.

    :param X: The positions of the distribution.
    :param mu: The weights of the distribution.
    :param alpha: The alpha parameter.
    :return: The partitioned samples.
    """
    if mu.dtype not in [torch.float32, torch.float64]:
        logging.raise_error(
            "The mu tensor must be of type float32 or float64", _log, ValueError
        )

    alpha = torch.tensor(alpha)

    if alpha <= 0:
        logging.raise_error(
            "The alpha parameter must be greater than 0", _log, ValueError
        )

    if _log.level <= logging.INFO:
        n_times = (
            torch.ceil(alpha * mu / torch.min(mu)).to(torch.int).to(mu.device)
        )
        _log.debug(f"Number of times to repeat each sample: {n_times}")
        n_rows = int(torch.sum(n_times))
        _log.info(f"Number of rows in the new X: {n_rows}")

    X, mu = _partition(X, mu, alpha)
    mu /= torch.sum(mu)

    return X, mu
