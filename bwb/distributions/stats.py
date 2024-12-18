import functools
from copy import copy
from typing import Callable, Protocol
from typing import Sequence as Seq

import PIL.Image
import torch
from matplotlib import pyplot as plt

import bwb.distributions.posterior_samplers as ps
import bwb.logging_ as logging
import bwb.utils as utils

__all__ = [
    "find_set_quantile",
    "order_distributions",
    "plot_log_like_models",
]

_log = logging.get_logger(__name__)


class _DistributionP(Protocol):
    """
    Protocol class for distributions.
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of the distribution.
        """


class _DistrDrawP(_DistributionP, Protocol):
    """
    Protocol class for distribution draws.
    """

    @property
    def image(self) -> PIL.Image.Image:
        """
        Return the image of the distribution draw.
        """


type _DistrT[DistrT: _DistributionP] = DistrT
type _DistrDrawT[DistrDrawT: _DistrDrawP] = DistrDrawT


def _cached_function_results() -> Callable:
    """
    Decorator to cache the results of a function.
    """

    # noinspection PyMissingTypeHints,PyMissingOrEmptyDocstring
    def wrapper[FuncT: Callable](func: FuncT) -> FuncT:
        cache = dict()

        # noinspection PyMissingTypeHints,PyMissingOrEmptyDocstring
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (
                tuple(id(a) for a in args),
                tuple((k, id(v)) for k, v in kwargs.items()),
            )
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        return wrapped

    return wrapper


@logging.register_total_time(_log)
@_cached_function_results()
def _log_like(
    distributions: Seq[_DistrT],
    data: torch.Tensor,
) -> Seq[float]:
    """
    Compute the log-likelihood of the models.
    """
    return [float(ps.log_likelihood_model(d, data)) for d in distributions]


@logging.register_total_time(_log)
def _determine_is_ordered(
    log_like_list: Seq[torch.Tensor | float],
    reverse: bool = False,
) -> bool:
    """
    Determine if the distributions are ordered.
    """
    log_like_tensor = torch.tensor(log_like_list)
    if reverse:
        return bool(torch.all(log_like_tensor[:-1] >= log_like_tensor[1:]))
    return bool(torch.all(log_like_tensor[:-1] <= log_like_tensor[1:]))


@logging.register_total_time(_log)
@_cached_function_results()
def order_distributions(
    distributions: Seq[_DistrT],
    data: torch.Tensor,
    reverse: bool = False,
) -> Seq[_DistrT]:
    """
    Order distributions by their mean value.
    """
    # Check if the distributions are already ordered
    log_like_list = _log_like(distributions, data)
    if _determine_is_ordered(log_like_list, reverse):
        _log.debug("Distributions are already ordered.")
        return distributions

    # If not, sort the distributions
    _log.debug("Sorting the distributions.")
    distr_ll = list(zip(distributions, log_like_list))
    sorted_ = sorted(distr_ll, key=lambda x: x[1], reverse=reverse)

    return [d for d, _ in sorted_]


@logging.register_total_time(_log, logging.INFO)
def plot_log_like_models(
    distributions: Seq[_DistrDrawT],
    data: torch.Tensor,
    reverse: bool = False,
    **kwargs,
) -> tuple[[plt.Figure, plt.Axes], Seq[_DistrDrawT]]:
    """
    Plot log-likelihood of models.
    """
    sorted_distr = order_distributions(distributions, data, reverse)
    log_likes = _log_like(sorted_distr, data)
    fig_ax = utils.plot_list_of_draws(
        list_of_draws=sorted_distr,
        labels=[
            r"$i={}$ $\ell = {:.2f}$".format(i, log_like)
            for i, log_like in enumerate(log_likes)
        ],
        **kwargs,
    )
    return fig_ax, sorted_distr


@logging.register_total_time(_log)
def _binary_search(
    sequence: Seq[float],
    key: float,
    reverse: bool = False,
) -> int:
    """
    Binary search for the key in the sequence. If the key is not found,
    return the index where the key should be inserted.
    """
    sequence = copy(sequence)

    if reverse:
        sequence = sequence[::-1]
    low, high = 0, len(sequence) - 1

    while low <= high:
        mid = (low + high) // 2
        if sequence[mid] < key:
            low = mid + 1
        elif sequence[mid] > key:
            high = mid - 1
        else:
            return mid

    return low


@logging.register_total_time(_log, logging.INFO)
def find_set_quantile(
    distributions: Seq[_DistrT],
    data: torch.Tensor,
    quantile: float,
    reverse: bool = False,
) -> Seq[_DistrT]:
    """
    Find the set of distributions that are within the quantile.
    """
    sorted_distr = order_distributions(distributions, data, reverse)
    log_like = torch.tensor(_log_like(sorted_distr, data))
    norm_like = torch.nn.functional.softmax(log_like, dim=0)
    cum_like = torch.cumsum(norm_like, dim=0)

    # Find index using binary search
    idx = _binary_search(cum_like, quantile, reverse)
    return sorted_distr[:idx]


if __name__ == "__main__":
    from random import random

    _log = logging.get_logger(__name__, logging.DEBUG)

    # Test the binary search
    seq_ = [random() for _ in range(10)]
    seq_.sort()
    key_ = 0.5
    idx_ = _binary_search(seq_, key_)
    print(f"Index of {key_} in {seq_} is {idx_}")
    print(seq_[:idx_], key_, seq_[idx_:])

    # Test ordered distributions
    seq_ = [random() for _ in range(10)]
    print(f"Is ordered: {_determine_is_ordered(seq_)}")
    print(f"Is ordered: {_determine_is_ordered(seq_, reverse=True)}")
    seq_.sort()
    print(f"Is ordered: {_determine_is_ordered(seq_)}")
    print(f"Is ordered: {_determine_is_ordered(seq_, reverse=True)}")
    seq_.reverse()
    print(f"Is ordered: {_determine_is_ordered(seq_)}")
    print(f"Is ordered: {_determine_is_ordered(seq_, reverse=True)}")
