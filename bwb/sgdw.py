import time
from typing import Callable, Union

import torch
from torch import linalg as LA

import bwb.distributions as distrib
import bwb.transports as tpt
from bwb import logging
from bwb import utils

__all__ = [
    "compute_bwb_distribution_draw",
]

_log = logging.get_logger(__name__)
_bar = "=" * 12


def compute_bwb_distribution_draw(
        transport: tpt.BaseTransport,
        posterior: distrib.PosteriorPiN[distrib.DiscreteDistribution],
        learning_rate: Callable[[int], float],  # The \gamma_k schedule
        batch_size: Union[Callable[[int], int], int],  # The S_k schedule
        alpha: float = 1.,
        tol: float = 1e-8,  # Tolerance to converge
        max_iter: int = 100_000,
        max_time: float = float("inf"),  # In seconds
        position_historial=False,  # Return the historial as a position
        distribution_historial=False,  # Return the distributions along the iterations
        samples_posterior_history=False,
):
    if isinstance(batch_size, int):
        aux = batch_size

        def batch_size(n):
            return aux
    batch_size: Callable[[int], int]

    # Paso 1: Muestrear un mu_0
    mu_0: distrib.DiscreteDistribution = posterior.draw()

    # Calcular las ubicaciones a través de la partición
    X_k, m = utils.partition(X=mu_0.enumerate_nz_support_(),
                             mu=mu_0.nz_probs,
                             alpha=alpha)
    _log.info(f"total init weights: {len(m)}")

    # Crear historiales
    if position_historial:
        position_historial = [X_k]
    if distribution_historial:
        distribution_historial = [distrib.DiscreteDistribution(support=X_k, weights=m)]
        pass
    if samples_posterior_history:
        samples_posterior_history = [[mu_0]]

    tic, toc = time.time(), time.time()
    diff_t = 0
    w_dist = float("inf")
    k = 0
    while (
            k < max_iter  # Alcanza iteración máxima
            and toc - tic < max_time  # Alcanza tiempo máximo
            and w_dist >= tol  # Alcanza convergencia en distancia
    ):
        tic_ = time.time()
        _log.info(_bar
                  + f" {k = }, "
                    f"t = {toc - tic:.4f} [seg], "
                    f"Δt = {diff_t * 1000:.4f} [ms], "
                    f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.4f} [ms/iter] " + _bar)

        if samples_posterior_history:
            samples_posterior_history.append([])
        T_X_k = torch.zeros_like(X_k)
        S_k = batch_size(k)
        for j in range(S_k):
            # Paso 2: Muestrear \tilde\mu^i_k
            t_mu_i_k: distrib.DiscreteDistribution = posterior.draw()
            if samples_posterior_history:
                samples_posterior_history[-1].append(t_mu_i_k)
            t_X_i_k, t_m_i_k = t_mu_i_k.enumerate_nz_support_(), t_mu_i_k.nz_probs

            # Calcular transporte óptimo
            transport.fit(
                Xs=X_k, mu_s=m,
                Xt=t_X_i_k, mu_t=t_m_i_k,
            )
            T_X_k += transport.transform(X_k)
        T_X_k /= S_k

        # Calcular la distribución de mu_{k+1}
        gamma_k = learning_rate(k)
        _log.debug(f"{gamma_k = :.6f}")
        X_kp1 = (1 - gamma_k) * X_k + gamma_k * T_X_k

        # Calcular la distancia de Wasserstein
        diff = X_k - T_X_k
        w_dist = float((gamma_k ** 2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        _log.debug(f"{w_dist = :.8f}")

        # Agregar a los historiales
        if position_historial:
            position_historial.append(X_kp1)
        if distribution_historial:
            distribution_historial.append(
                distrib.DiscreteDistribution(support=X_kp1, weights=m)
            )

        # Actualizar
        k += 1
        X_k = X_kp1
        toc = time.time()
        diff_t = toc - tic_

    to_return = [X_k, m]
    if position_historial:
        to_return.append(position_historial)
    if distribution_historial:
        to_return.append(distribution_historial)
    if samples_posterior_history:
        to_return.append(samples_posterior_history)

    return tuple(to_return)
