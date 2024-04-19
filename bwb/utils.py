"""
Module that contains utility functions.
"""
import collections as c
import functools
import time
import typing as t
import warnings

import ipyplot
import numpy as np
import PIL.Image
import torch

import bwb._logging as logging
from bwb.config import config
from bwb.exceptions import AutocorrError

__all__ = [
    "array_like_t",
    "device_t",
    "timeit_to_total_time",
    "set_generator",
    "freq_labels_dist_sampler",
    "grayscale_parser",
    "normalised_samples_ordered_dict",
    "partition",
    "function_1d",
    "integrated_time",
]

type array_like_t = np.ndarray | torch.Tensor | t.Iterable
type device_t = str | torch.device | int | None

_log = logging.get_logger(__name__)


@t.runtime_checkable
class DiscreteDistribSamplerP(t.Protocol):
    """
    Protocol for the distribution sampler.
    """
    samples_counter: c.Counter[int]


@t.runtime_checkable
class DrawP(t.Protocol):
    """
    Protocol for the draw.
    """
    image: PIL.Image.Image


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


# noinspection PyShadowingNames
def function_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D PyTorch tensor.

    Returns:
        tensor: The autocorrelation function of the time series.

    """
    x = torch.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Transfer input tensor to the appropriate device (e.g., GPU)
    device = x.device

    # Compute the FFT and then (from that) the auto-correlation function
    f = torch.fft.fft(x - torch.mean(x), n=2 * n)
    acf = torch.fft.ifft(f * torch.conj(f))[: len(x)].real
    acf /= acf[0].item()

    # Transfer the result back to the original device (if necessary)
    acf = acf.to(device)

    return acf


# noinspection PyShadowingNames,PyMissingOrEmptyDocstring
def auto_window(taus, c, device=None):
    if device is None:
        device = taus.device

    m = torch.arange(len(taus), device=device) < c * taus
    if torch.any(m):
        return torch.argmin(m.long()).item()
    return len(taus) - 1


# noinspection PyShadowingNames
def integrated_time(x, c=5, tol=50, quiet=True, has_walkers=True, device=None):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_ to
    determine a reasonable window size.

    Args:
        x (torch.Tensor): The time series. If 2-dimensional, the tensor
            dimensions are interpreted as ``(n_step, n_walker)`` unless
            ``has_walkers==False``, in which case they are interpreted as
            ``(n_step, n_param)``. If 3-dimensional, the dimensions are
            interpreted as ``(n_step, n_walker, n_param)``.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)
        has_walkers (Optional[bool]): Whether the last axis should be
            interpreted as walkers or parameters if ``x`` has 2 dimensions.
            (default: ``True``)
        device (Optional[torch.device]): The target device for computation.
            If not specified, the device of the input tensor `x` will be used.

    Returns:
        float or tensor: An estimate of the integrated autocorrelation time of
            the time series ``x``.

    Raises
        AutocorrError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.

    """
    # Check if a device is specified, otherwise use the device of the input tensor
    if device is None:
        device = x.device

    x = torch.atleast_1d(x.to(device))
    if len(x.shape) == 1:
        x = x[:, None, None]
    if len(x.shape) == 2:
        if not has_walkers:
            x = x[:, None, :]
        else:
            x = x[:, :, None]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = torch.empty(n_d, device=device)
    windows = torch.empty(n_d, dtype=torch.int, device=device)

    # Loop over parameters
    for d in range(n_d):
        f = torch.zeros(n_t, device=device)
        for k in range(n_w):
            f += function_1d(x[:, k, d])
        f /= n_w
        taus = 2.0 * torch.cumsum(f, dim=0) - 1.0
        taus = taus.to(device)
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if torch.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, torch.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
        if not quiet:
            raise AutocorrError(tau_est.cpu(), msg)
        _log.warning(msg)

    return tau_est


def timeit_to_total_time(method):
    """Function that records the total time it takes to execute a method, and stores it in the
    ``total_time`` attribute of the class instance."""

    # noinspection PyMissingOrEmptyDocstring
    @functools.wraps(method)
    def timeit_wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = method(*args, **kwargs)
        toc = time.perf_counter()
        args[0].total_time += toc - tic
        return result

    return timeit_wrapper


def set_generator(seed: int = None, device: device_t = "cpu") -> torch.Generator:
    """
    Set the generator for the random number generator.

    :param seed: The seed for the random number generator.
    :param device: The device to set the generator.
    :return: The generator.
    """
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
        return gen
    gen.manual_seed(seed)
    return gen


def freq_labels_dist_sampler(dist_sampler: DiscreteDistribSamplerP) -> t.Sequence[str]:
    """
    Function that returns the most common labels in the dist_sampler.

    :param dist_sampler: The distribution sampler to analyse.
    :return: A list of strings with the most common labels in the dist_sampler.
    """
    if not isinstance(dist_sampler, DiscreteDistribSamplerP):
        msg = "The dist_sampler must be an instance of PDiscreteDistribSampler."
        _log.error(msg)
        raise TypeError(msg)
    return [
        f"id: {id_},\nfreq: {freq}"
        for id_, freq in dist_sampler.samples_counter.most_common()
    ]


def normalised_samples_ordered_dict(posterior: DiscreteDistribSamplerP):
    """
    Function that returns an ordered dictionary with the normalised samples in the posterior instance.

    :param posterior: The posterior instance to analyse.
    :return: An ordered dictionary with the normalised samples in the MCMC instance.
    """
    counter = posterior.samples_counter
    return c.OrderedDict([(k, v / counter.total()) for k, v in counter.most_common()])


def __grayscale(
    to_return: torch.Tensor,
    weights: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    support1, support2 = support[:, 0], support[:, 1]
    for w, pos1, pos2 in zip(weights, support1, support2):
        to_return[pos1, pos2] += w
    to_return = (to_return / torch.max(to_return) * 255)  # .type(torch.uint8)
    return to_return


# noinspection PyUnreachableCode
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
    device = config.device if device is None else device
    dtype = config.dtype if dtype is None else dtype
    support = torch.round(support).type(torch.int32)
    to_return = torch.zeros(shape, dtype=dtype, device=device)
    to_return: torch.Tensor = _grayscale(to_return, weights, support)

    return to_return.type(torch.uint8)


# noinspection PyPep8Naming
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


# noinspection PyPep8Naming
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
        raise ValueError("The alpha parameter must be greater than 0")

    if _log.level <= logging.INFO:
        n_times = torch.ceil(alpha * mu / torch.min(mu)).type(torch.int).to(mu.device)
        _log.debug(f"Number of times to repeat each sample: {n_times}")
        n_rows = int(torch.sum(n_times))
        _log.info(f"Number of rows in the new X: {n_rows}")

    X, mu = _partition(X, mu, alpha)
    mu = mu / torch.sum(mu)

    return X, mu


# ================= DEPRECATED FUNCTIONS =================
def plot_list_of_images(list_of_images: t.Sequence[PIL.Image.Image], **kwargs):
    """
    Function that plots a list of images.

    :param list_of_images: The list of images to draw.
    :param kwargs: Optional arguments to pass to the ipyplot.plot_images function. For further
        information, please see the documentation of that function.
    """
    msg = ("The plot_list_of_images function is deprecated and will be removed in the future. "
           "Use plotters.plot_list_of_images instead.")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Set values by default
    kwargs.setdefault("max_images", 36)
    kwargs.setdefault("img_width", 75)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipyplot.plot_images(list_of_images, **kwargs)


def plot_list_of_draws(list_of_draws: t.Sequence[PDraw], **kwargs):
    """
    Function that plots a list of DistributionDraws instances.

    :param list_of_draws: The list of distributions to draw.
    :param kwargs: Optional arguments to pass to the ipyplot.plot_images function. For further
        information, please see the documentation of that function.
    """
    msg = ("The plot_list_of_draws function is deprecated and will be removed in the future. "
           "Use plotters.plot_list_of_draws instead.")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return plot_list_of_images([draw.image for draw in list_of_draws], **kwargs)


def likelihood_ordered_dict(posterior):
    """
    Function that returns an ordered dictionary with the likelihoods of the samples in the posterior.

    :param posterior: The posterior to analyse.
    :return: An ordered dictionary with the likelihoods of the samples in the posterior.
    """
    msg = "The likelihood_ordered_dict function is deprecated and will be removed in the future. "
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    like_cache = posterior.likelihood_cache
    posterior_probs = like_cache / np.sum(like_cache)
    likelihood_dct = c.OrderedDict({i: prob for i, prob in enumerate(posterior_probs)})

    for key, _ in sorted(likelihood_dct.items(), key=lambda item: -item[1]):
        likelihood_dct.move_to_end(key)

    return likelihood_dct


def normalised_steps_ordered_dict(mcmc):
    """
    Function that returns an ordered dictionary with the normalised steps in the MCMC instance.

    :param mcmc: The MCMC instance to analyse.
    :return: An ordered dictionary with the normalised steps in the MCMC instance.
    """
    msg = "The normalised_steps_ordered_dict function is deprecated and will be removed in the future."
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    counter = mcmc.steps_counter
    return c.OrderedDict([(k, v / counter.total()) for k, v in counter.most_common()])


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
