"""
This module provides functions to estimate the autocorrelation time of
a time series.
"""
import torch

import bwb._logging as logging
from bwb.exceptions import AutocorrError

__all__ = [
    "function_1d",
    "integrated_time",
]

_log = logging.get_logger(__name__)


def next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


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


def auto_window(taus, c, device=None):
    """Estimate the window size for the autocorrelation function."""
    if device is None:
        device = taus.device

    m = torch.arange(len(taus), device=device) < c * taus
    if torch.any(m):
        return torch.argmin(m.long()).item()
    return len(taus) - 1


def integrated_time(x, c=5, tol=50, quiet=True, has_walkers=True, device=None):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes
    <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_
    to determine a reasonable window size.

    Args:
        x (torch.Tensor): The time series. If 2-dimensional, the tensor
            dimensions are interpreted as ``(n_step, n_walker)`` unless
            ``has_walkers==False``, in which case they are interpreted
            as ``(n_step, n_param)``. If 3-dimensional, the dimensions
            are interpreted as ``(n_step, n_walker, n_param)``.
        c (Optional[float]): The step size for the window search.
            (default: ``5``)
        tol (Optional[float]): The minimum number of autocorrelation
            times needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when
            the chain is too short. If ``True``, give a warning instead
            of raising an :class:`AutocorrError`. (default: ``False``)
        has_walkers (Optional[bool]): Whether the last axis should be
            interpreted as walkers or parameters if ``x`` has 2
            dimensions. (default: ``True``)
        device (Optional[torch.device]): The target device for
            computation. If not specified, the device of the input
            tensor `x` will be used.

    Returns:
        float or tensor: An estimate of the integrated autocorrelation
            time of the time series ``x``.

    Raises
        AutocorrError: If the autocorrelation time can't be reliably
            estimated from the chain and ``quiet`` is ``False``. This
            normally means
            that the chain is too short.

    """
    # Check if a device is specified, otherwise use the device of the
    #   input tensor
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
