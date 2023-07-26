import time
from typing import Callable, Union

import torch
from torch import linalg as LA
from bwb import bregman

import bwb.distributions as distrib
import bwb.transports as tpt
from bwb import logging
from bwb import utils

__all__ = [
    "compute_bwb_discrete_distribution",
]

_log = logging.get_logger(__name__)
_bar = "=" * 12


def compute_bwb_discrete_distribution(
        transport: tpt.BaseTransport,
        posterior: distrib.PosteriorPiN[distrib.DiscreteDistribution],
        learning_rate: Callable[[int], float],  # The \gamma_k schedule
        batch_size: Union[Callable[[int], int], int],  # The S_k schedule
        alpha: float = 1.,
        tol: float = 1e-8,  # Tolerance to converge
        max_iter: int = 100_000,
        max_time: float = float("inf"),  # In seconds
        position_history=False,  
        distribution_history=False, 
        samples_posterior_history=False,
):
    if isinstance(batch_size, int):
        aux = batch_size

        def batch_size(n):
            return aux
    batch_size: Callable[[int], int]

    # Paso 1: Sampling a mu_0
    mu_0: distrib.DiscreteDistribution = posterior.draw()
    dtype, device = mu_0.dtype, mu_0.device
    _log.info(f"{dtype = }, {device = }")

    # Calculate locations through the partition
    X_k, m = utils.partition(
        X=mu_0.enumerate_nz_support_(), 
        mu=mu_0.nz_probs,
        alpha=alpha
        )
    _log.info(f"total init weights: {len(m)}")

    # Create histories
    if position_history:
        position_history = [X_k]
    if distribution_history:
        distribution_history = [distrib.DiscreteDistribution(support=X_k, weights=m)]
    if samples_posterior_history:
        samples_posterior_history = [[mu_0]]

    tic, toc = time.time(), time.time()
    diff_t = 0
    w_dist = float("inf")
    k = 0
    while (
            k < max_iter  # Reaches maximum iteration
            and toc - tic < max_time  # Reaches maximum time
            and w_dist >= tol  # Achieves convergence in distance
    ):
        tic_ = time.time()
        _log.info(
            _bar 
            + f" {k = }, "
            f"t = {toc - tic:.4f} [seg], "
            f"Δt = {diff_t * 1000:.4f} [ms], "
            f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.4f} [ms/iter] " + _bar
            )

        if samples_posterior_history:
            samples_posterior_history.append([])

        T_X_k = torch.zeros_like(X_k, dtype=dtype, device=device)
        S_k = batch_size(k)
        for _ in range(S_k):
            # Paso 2: Draw \tilde\mu^i_k
            t_mu_i_k: distrib.DiscreteDistribution = posterior.draw()
            if samples_posterior_history:
                samples_posterior_history[-1].append(t_mu_i_k)
            t_X_i_k, t_m_i_k = t_mu_i_k.enumerate_nz_support_(), t_mu_i_k.nz_probs
            t_m_i_k /= torch.sum(t_m_i_k)

            # Calculate optimal transport
            transport.fit(
                Xs=X_k, mu_s=m,
                Xt=t_X_i_k, mu_t=t_m_i_k,
            )
            T_X_k += transport.transform(X_k)
        T_X_k /= S_k

        # Calculate the distribution of mu_{k+1}
        gamma_k = learning_rate(k)
        _log.debug(f"{gamma_k = :.6f}")
        X_kp1 = (1 - gamma_k) * X_k + gamma_k * T_X_k

        # Calculate Wasserstein distance
        diff = X_k - T_X_k
        w_dist = float((gamma_k ** 2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        _log.debug(f"{w_dist = :.8f}")

        # Add to history
        if position_history:
            position_history.append(X_kp1)
        if distribution_history:
            distribution_history.append(
                distrib.DiscreteDistribution(support=X_kp1, weights=m)
            )

        # Update
        k += 1
        X_k = X_kp1
        toc = time.time()
        diff_t = toc - tic_

    to_return = [X_k, m]
    if position_history:
        to_return.append(position_history)
    if distribution_history:
        to_return.append(distribution_history)
    if samples_posterior_history:
        to_return.append(samples_posterior_history)

    return tuple(to_return)


def compute_bwb_distribution_draw(
        posterior: distrib.PosteriorPiN[distrib.DistributionDraw],
        learning_rate: Callable[[int], float],  # The \gamma_k schedule
        reg: float = 3e-3,  # Regularization of the convolutional method
        max_iter: int = 100_000,
        max_time: float = float("inf"),  # In seconds
        weights_history=False,  
        distribution_history=False, 
        samples_posterior_history=False,
):

    # Paso 1: Sampling a mu_0
    mu_k: distrib.DistributionDraw = posterior.draw()
    dtype, device = mu_k.dtype, mu_k.device
    _log.info(f"{dtype = }, {device = }")

    gs_weights_k = mu_k.grayscale_weights

    # Create histories
    if weights_history:
        weights_history = [gs_weights_k]
    if distribution_history:
        distribution_history = [mu_k]
    if samples_posterior_history:
        samples_posterior_history = [[mu_k]]

    tic, toc = time.time(), time.time()
    diff_t = 0
    k = 0
    while (
            k < max_iter  # Reaches maximum iteration
            and toc - tic < max_time  # Reaches maximum time
    ):
        tic_ = time.time()
        _log.info(
            _bar 
            + f" {k = }, "
            f"t = {toc - tic:.4f} [seg], "
            f"Δt = {diff_t * 1000:.4f} [ms], "
            f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.4f} [ms/iter] " + _bar
            )

        m_k: distrib.DistributionDraw = posterior.draw()
        if samples_posterior_history:
            samples_posterior_history.append([m_k])

        # Calculate the distribution of mu_{k+1}
        gamma_k = learning_rate(k)
        _log.debug(f"{gamma_k = :.6f}")

        gs_weights_kp1, _ = bregman.convolutional_barycenter2d(
            A=[gs_weights_k, m_k.grayscale_weights],
            reg=reg, weights=[1-gamma_k, gamma_k],
            numItermax=1_000, stopThr=1e-8,
            warn=False, log=True,
        )

        # Add to history
        if weights_history:
            weights_history.append(gs_weights_k)
        if distribution_history:
            mu_kp1 = distrib.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
            distribution_history.append(mu_kp1)

        # Update
        k += 1
        gs_weights_k = gs_weights_kp1
        toc = time.time()
        diff_t = toc - tic_

    mu = distrib.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
    to_return = [mu]
    if weights_history:
        to_return.append(weights_history)
    if distribution_history:
        to_return.append(distribution_history)
    if samples_posterior_history:
        to_return.append(samples_posterior_history)

    return tuple(to_return)

