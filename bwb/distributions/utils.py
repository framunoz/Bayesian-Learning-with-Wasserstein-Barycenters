import torch

import bwb._logging as logging
from bwb.config import config

__all__ = [
    "grayscale_parser",
    "partition",
]

_log = logging.get_logger(__name__)


def __grayscale(
    to_return: torch.Tensor,
    weights: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    support1, support2 = support[:, 0], support[:, 1]
    for w, pos1, pos2 in zip(weights, support1, support2):
        to_return[pos1, pos2] += w
    to_return = to_return / torch.max(to_return)  # .type(torch.uint8)
    to_return = to_return * 255
    return to_return


_grayscale = torch.jit.script(
    __grayscale,
    example_inputs=[(
        torch.rand((28, 28), dtype=torch.float32, device=config.device),  # to_return
        torch.rand((784,), dtype=torch.float32, device=config.device),  # weights
        torch.randint(0, 28, size=(784, 2), dtype=torch.int32, device=config.device),  # support
    ), (
        torch.rand((28, 28), dtype=torch.float64, device=config.device),  # to_return
        torch.rand((784,), dtype=torch.float64, device=config.device),  # weights
        torch.randint(0, 28, size=(784, 2), dtype=torch.int32, device=config.device),  # support
    )]
)


def grayscale_parser(
    shape: tuple[int, ...],
    weights: torch.Tensor,
    support: torch.Tensor,
    device=None,
    dtype=None
) -> torch.Tensor:
    """
    Function that parses the weights and support into a grayscale image.

    :param shape: The shape of the image.
    :param weights: The weights of the distribution.
    :param support: The support of the distribution.
    :param device: The device to use.
    :param dtype: The dtype to use.
    :return: A grayscale image.
    """
    device = weights.device
    dtype = weights.dtype
    support = torch.round(support).to(dtype=torch.int32, device=device)
    to_return = torch.zeros(shape, dtype=dtype, device=device)
    to_return: torch.Tensor = _grayscale(to_return, weights, support)

    return to_return.type(torch.uint8)


@torch.jit.script
def _partition(
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


def partition(X: torch.Tensor, mu: torch.Tensor, alpha: float):
    """
    Function that partitions the samples in X according to the weights in mu.

    :param X: The positions of the distribution.
    :param mu: The weights of the distribution.
    :param alpha: The alpha parameter.
    :return: The partitioned samples.
    """
    alpha = torch.tensor(alpha)

    if alpha <= 0:
        logging.raise_error(
            "The alpha parameter must be greater than 0",
            _log, ValueError
        )

    if _log.level <= logging.INFO:
        n_times = torch.ceil(alpha * mu / torch.min(mu)).type(torch.int).to(mu.device)
        _log.debug(f"Number of times to repeat each sample: {n_times}")
        n_rows = int(torch.sum(n_times))
        _log.info(f"Number of rows in the new X: {n_rows}")

    X, mu = _partition(X, mu, alpha)
    mu = mu / torch.sum(mu)

    return X, mu


def main():
    """
    Main function.
    """
    shape = (28, 28)

    # Test 1
    weights = torch.rand((784,), device=config.device)
    support = torch.randint(0, 28, (784, 2), device=config.device)
    to_return = grayscale_parser(shape, weights, support)
    print(to_return.shape)
    print(to_return.dtype)
    print(to_return.device)

    # Test 2
    weights = torch.rand((784,), device=config.device)
    weights = weights / torch.sum(weights)
    support = torch.rand((784, 2), device=config.device) * 27
    print(support)
    to_return = grayscale_parser(shape, weights, support)
    print(to_return.shape)
    print(to_return.dtype)
    print(to_return.device)


if __name__ == '__main__':
    main()
