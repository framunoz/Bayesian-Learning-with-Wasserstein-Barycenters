"""
Module for Stochastic Gradient Descent Algorithms in Wasserstein Space.
"""
import abc
import time
import typing as t
import warnings
from copy import deepcopy

import ot
import torch
from torch import linalg as LA

import bwb.distributions as D
import bwb.distributions.utils as D_utils
import bwb.logging_ as logging
import bwb.pot.bregman as bregman
import bwb.pot.transports as tpt
from . import utils
from .. import protocols as P

__all__ = [
    "Runnable",
    "SGDW",
    "BaseSGDW",
    "DiscreteDistributionSGDW",
    "DistributionDrawSGDW",
    "ConvDistributionDrawSGDW",
    "DebiesedDistributionDrawSGDW",
]

type DiscretePosWgt = tuple[torch.Tensor, torch.Tensor]
type CallbackFn = t.Callable[[], None]

_log = logging.get_logger(__name__)
_bar = "=" * 5


class Runnable[DistributionT](metaclass=abc.ABCMeta):
    """
    Interface for classes that can be run.
    """

    @abc.abstractmethod
    def run(self) -> DistributionT:
        """
        Run the algorithm.
        """
        ...


class DistributionSamplerP[DistributionT](P.HasDeviceDTypeP, t.Protocol):
    """
    A protocol for distribution samplers.
    """

    def sample(self, n: int) -> t.Sequence[DistributionT]:
        """
        Sample a sequence of distributions.

        :param n: The number of distributions to sample.
        :return: The sequence of distributions.
        """
        ...

    def draw(self) -> DistributionT:
        """
        Draw a distribution.

        :return: The distribution.
        """
        ...


class SGDW[DistributionT, PosWgtT](
    Runnable[DistributionT],
    P.HasDeviceDType,
    metaclass=abc.ABCMeta
):
    r"""
    Base class for Stochastic Gradient Descent in Wasserstein Space.
    """
    distr_sampler: DistributionSamplerP[DistributionT]
    schd: utils.Schedule
    det_params: utils.DetentionParameters
    iter_params: utils.IterationParameters

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[DistributionT],
        schd: utils.Schedule,
        det_params: utils.DetentionParameters,
        iter_params: utils.IterationParameters,
    ):
        self.distr_sampler = distr_sampler
        self.schd = schd
        self.det_params = det_params
        self.iter_params = iter_params

    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        """
        Additional representation for the class.
        """
        return ""

    def _repr_(self, sep: str, tab: str, n_tab: int, new_line: str) -> str:
        """
        Representation for the class.
        """
        to_return = tab * n_tab + self.__class__.__name__ + "("
        add_repr = self._additional_repr_(sep, tab, n_tab + 1, new_line)
        if add_repr:
            to_return += new_line + add_repr + tab * n_tab
        to_return += ")"

        return to_return

    def __repr__(self) -> str:
        new_line = "\n"
        tab = "  "
        comma = ","
        sep = comma + new_line

        return self._repr_(sep, tab, 0, new_line)

    @t.final
    @t.override
    @property
    def dtype(self) -> torch.dtype:
        return self.distr_sampler.dtype

    @t.final
    @t.override
    @property
    def device(self) -> torch.device:
        return self.distr_sampler.device

    @property
    @abc.abstractmethod
    def callback(self) -> CallbackFn:
        """
        Callback function to run at the end of each iteration.
        """
        ...

    @callback.setter
    def callback(self, callback: CallbackFn) -> None:
        """
        Set the callback function to run at the end of each iteration.
        """
        ...

    @abc.abstractmethod
    def first_sample(self) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        """
        Draw the first sample from the distribution sampler.
        """
        ...

    def init_algorithm(self) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        """
        Initialize the algorithm.
        """
        # Step 1: Sampling a mu_0
        lst_mu_0, pos_wgt_0 = self.first_sample()

        return lst_mu_0, pos_wgt_0

    @abc.abstractmethod
    def update_pos_wgt(
        self,
        pos_wgt_k: PosWgtT,
        lst_mu_k: t.Sequence[DistributionT],
        gamma_k: float,
    ) -> PosWgtT:
        """
        Update the position and weight for the next iteration.
        """
        ...

    @abc.abstractmethod
    def _compute_wass_dist(
        self, pos_wgt_k: PosWgtT, pos_wgt_kp1: PosWgtT, gamma_k: float
    ) -> float:
        """
        Compute the Wasserstein distance between two positions and weights.
        """
        ...

    def compute_wass_dist(
        self, pos_wgt_k: PosWgtT, pos_wgt_kp1: PosWgtT, gamma_k: float
    ) -> float:
        """
        Compute the Wasserstein distance between two positions and weights.
        """
        wass_dist = self._compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)
        return self.iter_params.update_wass_dist(wass_dist)

    def step_algorithm(
        self, k: int, pos_wgt_k: PosWgtT
    ) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        """
        Run a step of the algorithm.
        """
        # Step 2: Draw S_k samples from the distribution sampler
        S_k = self.schd.batch_size(k)
        lst_mu_k = self.distr_sampler.sample(S_k)

        # Step 3: Compute the distribution of mu_{k+1}
        gamma_k = self.schd.step_schedule(k)
        pos_wgt_kp1 = self.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

        # Step 4 (optional): Compute the Wasserstein distance
        self.compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)

        return lst_mu_k, pos_wgt_kp1

    @abc.abstractmethod
    def get_pos_wgt(self, mu: DistributionT) -> PosWgtT:
        """
        Get the position and weight from a distribution.
        """
        ...

    @abc.abstractmethod
    def create_distribution(self, pos_wgt: PosWgtT) -> DistributionT:
        """
        Create a distribution from the position and weight.
        """
        ...

    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        """
        Create the barycenter from the position and weight.
        """
        return self.create_distribution(pos_wgt)

    @t.final
    @t.override
    def run(self) -> DistributionT:
        """
        Run the algorithm.
        """
        _, pos_wgt_k = self.init_algorithm()

        # The logic of the detention criteria and update are in the
        #   iterable class :class:`IterationParameters`
        for k in self.iter_params:
            # Run a step of the algorithm
            _, pos_wgt_k = self.step_algorithm(k, pos_wgt_k)

            # Callback to do extra instructions at the end of each iteration
            self.callback()

        barycenter = self.create_barycenter(pos_wgt_k)

        return barycenter


# MARK: BaseSGDW Class
# noinspection PyMethodOverriding
class BaseSGDW[DistributionT, PosWgtT](
    SGDW[DistributionT, PosWgtT],
    metaclass=abc.ABCMeta
):
    r"""
    Base class for Stochastic Gradient Descent in Wasserstein Space.

    This class provides a base implementation for Stochastic Gradient
    Descent in Wasserstein Space. It defines the common attributes and
    methods used by the derived classes.

    :param step_scheduler: A callable function that takes an integer
        argument :math:`k` and returns the learning rate :math:`\gamma_k`
        for iteration :math:`k`.
    :type step_scheduler: callable
    :param batch_size: A callable function that takes an integer
        argument :math:`k` and returns the batch size :math:`S_k` for
        iteration :math:`k`. Alternatively, it can be a constant integer
        value.
    :type batch_size: callable or int
    :param tol: The tolerance value for convergence. Defaults to 1e-8.
    :type tol: float
    :param max_iter: The maximum number of iterations.
        Defaults to 100_000.
    :type max_iter: int
    :param max_time: The maximum time allowed for the algorithm to run.
        Defaults to infinity.
    :type max_time: float

    :raises TypeError: If ``learning_rate`` is not a callable or
        ``batch_size`` is not a callable or an integer.
    :raises ValueError: If ``learning_rate`` does not return a float or
        ``batch_size`` does not return an integer.
    """

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[DistributionT],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg = 1,
        tol: float = 1e-8,  # Tolerance to converge
        max_iter: int = 100_000,  # Maximum number of iterations
        max_time: float = float("inf"),  # Maximum time in seconds
    ):
        # Schedule parameters
        schd = utils.Schedule(step_scheduler, batch_size)

        # Detention parameters
        det_params = utils.DetentionParameters(tol, max_iter, max_time)

        # Iteration metrics
        iter_params = utils.IterationParameters(det_params)

        super().__init__(distr_sampler, schd, det_params, iter_params)

        # Callback
        self._callback: CallbackFn = lambda: None

        # A value to pass to the device and dtype
        self._val = torch.tensor(1, dtype=self.dtype, device=self.device)

    @property
    def callback(self) -> CallbackFn:
        """
        Callback function to run at the end of each iteration.

        :return: The callback function.
        """
        return self._callback

    @callback.setter
    def callback(self, callback: CallbackFn) -> None:
        self._callback = callback

    def _compute_wass_dist(
        self, pos_wgt_k: PosWgtT, pos_wgt_kp1: PosWgtT, gamma_k: float
    ) -> float:
        """
        Compute the Wasserstein distance between two positions and
        weights.

        This method should compute the Wasserstein distance between two
        positions and weights and return it.

        :param pos_wgt_k: The position and weight that come from the
            current sample.
        :param pos_wgt_kp1: The position and weight that come from the
            next sample.
        :param gamma_k: The learning rate for the next sample.
        """
        return float("inf")


# MARK: SGDW with discrete distributions
class DiscreteDistributionSGDW(
    BaseSGDW[D.DiscreteDistribution, DiscretePosWgt]
):
    def __init__(
        self,
        transport: tpt.BaseTransport,
        distr_sampler: DistributionSamplerP[D.DiscreteDistribution],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg = 1,
        alpha: float = 1.0,
        tol: float = 1e-8,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            step_scheduler=step_scheduler,
            batch_size=batch_size,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        self.transport = transport
        self.alpha = alpha
        self.include_w_dist = True

    @t.final
    @t.override
    def create_distribution(
        self, pos_wgt: DiscretePosWgt
    ) -> D.DiscreteDistribution:
        X_k, m = pos_wgt
        return D.DiscreteDistribution(support=X_k, weights=m)

    @t.final
    @t.override
    def get_pos_wgt(self, mu: D.DiscreteDistribution) -> DiscretePosWgt:
        return mu.enumerate_nz_support_(), mu.nz_probs

    @t.final
    @t.override
    def first_sample(
        self,
    ) -> tuple[t.Sequence[D.DiscreteDistribution], DiscretePosWgt]:
        mu_0: D.DiscreteDistribution = self.distr_sampler.draw()
        X_k, m = D_utils.partition(
            X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=self.alpha
        )
        X_k, m = X_k.to(self._val), m.to(self._val)
        return [mu_0], (X_k, m)

    @t.final
    @t.override
    def update_pos_wgt(
        self,
        pos_wgt_k: DiscretePosWgt,
        lst_mu_k: t.Sequence[D.DiscreteDistribution],
        gamma_k: float,
    ) -> DiscretePosWgt:
        X_k, m = pos_wgt_k
        T_X_k: torch.Tensor = torch.zeros_like(
            X_k, dtype=self.dtype, device=self.device
        )
        S_k = len(lst_mu_k)
        for mu_i_k in lst_mu_k:
            X_i_k, m_i_k = self.get_pos_wgt(mu_i_k)
            X_i_k, m_i_k = X_i_k.to(self._val), m_i_k.to(self._val)
            m_i_k /= torch.sum(m_i_k)
            self.transport.fit(Xs=X_k, mu_s=m, Xt=X_i_k, mu_t=m_i_k)
            T_X_k += self.transport.transform(X_k)
        T_X_k /= S_k
        # noinspection PyTypeChecker
        X_kp1: torch.Tensor = (1 - gamma_k) * X_k + gamma_k * T_X_k
        return X_kp1, m

    @t.final
    @t.override
    def _compute_wass_dist(
        self,
        pos_wgt_k: DiscretePosWgt,
        pos_wgt_kp1: DiscretePosWgt,
        gamma_k: float
    ):
        X_k, m = pos_wgt_k
        X_kp1, m = pos_wgt_kp1
        diff = X_k - X_kp1
        w_dist = float((gamma_k ** 2)
                       * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        return w_dist


# MARK: SGDW with distributions based in draws
class DistributionDrawSGDW(
    BaseSGDW[D.DistributionDraw, torch.Tensor], metaclass=abc.ABCMeta
):
    _conv_bar_kwargs = dict(
        reg=3e-3,
        method="sinkhorn",
        numItermax=10_000,
        stopThr=1e-4,
        verbose=False,
        warn=False,
    )

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[D.DistributionDraw],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg = 1,
        tol: float = 0,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            step_scheduler=step_scheduler,
            batch_size=batch_size,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        self.conv_bar_kwargs = deepcopy(self._conv_bar_kwargs)

    @t.final
    @t.override
    def create_distribution(
        self,
        pos_wgt: torch.Tensor
    ) -> D.DistributionDraw:
        gs_weights_k = pos_wgt
        return D.DistributionDraw.from_grayscale_weights(gs_weights_k)

    @t.final
    @t.override
    def get_pos_wgt(self, mu: D.DistributionDraw) -> torch.Tensor:
        return mu.grayscale_weights

    @t.final
    @t.override
    def first_sample(
        self,
    ) -> tuple[t.Sequence[D.DistributionDraw], torch.Tensor]:
        mu_0: D.DistributionDraw = self.distr_sampler.draw()
        return [mu_0], mu_0.grayscale_weights

    def set_geodesic_params(
        self,
        reg=3e-3,
        method="sinkhorn",
        num_iter_max=10_000,
        stop_thr=1e-4,
        verbose=False,
        warn=False,
        **kwargs,
    ) -> t.Self:
        """
        Set the parameters for geodesic computation.

        :param float reg: Regularization term for Sinkhorn algorithm.
            Default is 3e-3.
        :param str method: Method to use for geodesic computation.
            Default is "sinkhorn".
        :param int num_iter_max: Maximum number of iterations for
            Sinkhorn algorithm. Default is 1000.
        :param float stop_thr: Stopping threshold for Sinkhorn
            algorithm. Default is 1e-8.
        :param bool verbose: Whether to print verbose output during
            computation. Default is False.
        :param bool warn: Whether to display warning messages.
            Default is False.
        :param kwargs: Additional keyword arguments to be passed to
            the geodesic computation method.
        :return: The current instance of the class.
        """
        self.conv_bar_kwargs.update(
            reg=reg,
            method=method,
            numItermax=num_iter_max,
            stopThr=stop_thr,
            verbose=verbose,
            warn=warn,
            **kwargs,
        )

        return self

    @abc.abstractmethod
    def _compute_geodesic(
        self,
        gs_weights_lst_k: list[torch.Tensor],
        lst_gamma_k: list[float],
    ) -> torch.Tensor:
        """
        Compute the geodesic between two points in the Wasserstein space.

        :param gs_weights_lst_k: The list of grayscale weights.
        :param lst_gamma_k: The list of weights for the geodesic.
        :return: The geodesic between the points.
        """
        ...

    @t.final
    @t.override
    def update_pos_wgt(
        self,
        pos_wgt_k: torch.Tensor,
        lst_mu_k: t.Sequence[D.DistributionDraw],
        gamma_k: float
    ) -> torch.Tensor:
        S_k = len(lst_mu_k)
        gs_weights_kp1 = self._compute_geodesic(
            [pos_wgt_k] + [mu_k.grayscale_weights for mu_k in lst_mu_k],
            [1 - gamma_k] + [gamma_k / S_k] * S_k
        )
        return gs_weights_kp1


class ConvDistributionDrawSGDW(DistributionDrawSGDW):
    @t.final
    @t.override
    def _compute_geodesic(self, gs_weights_lst_k, lst_gamma_k) -> torch.Tensor:
        return bregman.convolutional_barycenter2d(
            A=torch.stack(gs_weights_lst_k),
            weights=torch.as_tensor(lst_gamma_k,
                                    dtype=self.dtype, device=self.device),
            **self.conv_bar_kwargs,
        )


class DebiesedDistributionDrawSGDW(DistributionDrawSGDW):
    _conv_bar_kwargs = dict(
        reg=1e-2,
        method="sinkhorn",
        numItermax=10_000,
        stopThr=1e-3,
        verbose=False,
        warn=False,
    )

    @t.final
    @t.override
    def _compute_geodesic(self, gs_weights_lst_k, lst_gamma_k) -> torch.Tensor:
        return ot.bregman.convolutional_barycenter2d_debiased(
            A=torch.stack(gs_weights_lst_k),
            weights=torch.as_tensor(lst_gamma_k,
                                    dtype=self.dtype, device=self.device),
            **self.conv_bar_kwargs,
        )

    @t.final
    @t.override
    def set_geodesic_params(
        self,
        reg=1e-2,
        method="sinkhorn",
        num_iter_max=10_000,
        stop_thr=1e-3,
        verbose=False,
        warn=False,
        **kwargs,
    ) -> t.Self:
        return super().set_geodesic_params(
            reg=reg,
            method=method,
            num_iter_max=num_iter_max,
            stop_thr=stop_thr,
            verbose=verbose,
            warn=warn,
            **kwargs,
        )


# MARK: Deprecated functions
# noinspection PyMissingOrEmptyDocstring
def compute_bwb_discrete_distribution(
    transport: tpt.BaseTransport,
    distrib_sampler: DistributionSamplerP[D.DiscreteDistribution],
    learning_rate: t.Callable[[int], float],  # The \gamma_k schedule
    batch_size: t.Union[t.Callable[[int], int], int],  # The S_k schedule
    alpha: float = 1.0,
    tol: float = 1e-8,  # Tolerance to converge
    max_iter: int = 100_000,
    max_time: float = float("inf"),  # In seconds
    position_history=False,
    distribution_history=False,
    distrib_sampler_history=False,
    report_every=10,
):
    # Warning for deprecation
    warnings.warn(
        "This function is deprecated. "
        "Use the DiscreteDistributionSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(batch_size, int):
        aux = batch_size

        # noinspection PyMissingOrEmptyDocstring,PyUnusedLocal
        def batch_size(n: int) -> int:
            return aux

    batch_size: t.Callable[[int], int]

    # Paso 1: Sampling a mu_0
    mu_0: D.DiscreteDistribution = distrib_sampler.draw()
    dtype, device = mu_0.dtype, mu_0.device

    # Compute locations through the partition
    _log.info("Computing initial weights")
    tic = time.time()
    X_k, m = D_utils.partition(
        X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=alpha
    )
    X_k, m = X_k.to(dtype=dtype, device=device), m.to(dtype=dtype,
                                                      device=device)
    toc = time.time()
    _log.info(f"total init weights: {len(m)}, Δt = {toc - tic:.2f} [seg]")

    # Create histories
    if position_history:
        position_history = [X_k]
    if distribution_history:
        distribution_history = [
            D.DiscreteDistribution(support=X_k, weights=m)]
    if distrib_sampler_history:
        distrib_sampler_history = [[mu_0]]

    tic, toc = time.time(), time.time()  # Start time
    diff_t = 0  # Time difference
    w_dist = float("inf")  # Wasserstein distance
    k = 0  # Iteration counter
    while (
        k < max_iter  # Reaches maximum iteration
        and toc - tic < max_time  # Reaches maximum time
        and w_dist >= tol  # Achieves convergence in distance
    ):
        tic_ = time.time()

        if k % report_every == 0:
            _log.info(
                _bar
                + f" k = {k}, "
                + f"w_dist = {w_dist:.4f}, "
                + f"t = {toc - tic:.2f} [seg], "
                + f"Δt = {diff_t * 1000:.2f} [ms], "
                + f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} "
                  f"[ms/iter] "
                + _bar
            )

        if distrib_sampler_history:
            distrib_sampler_history.append([])

        T_X_k = torch.zeros_like(X_k, dtype=dtype, device=device)
        S_k = batch_size(k)
        for _ in range(S_k):
            # Paso 2: Draw \tilde\mu^i_k
            t_mu_i_k: D.DiscreteDistribution = distrib_sampler.draw()
            if distrib_sampler_history:
                distrib_sampler_history[-1].append(t_mu_i_k)
            t_X_i_k = t_mu_i_k.enumerate_nz_support_()
            t_m_i_k = t_mu_i_k.nz_probs
            # Pass to device
            t_X_i_k = t_X_i_k.to(dtype=dtype, device=device)
            t_m_i_k = t_m_i_k.to(dtype=dtype, device=device)
            t_m_i_k /= torch.sum(t_m_i_k)

            # Compute optimal transport
            transport.fit(
                Xs=X_k,
                mu_s=m,
                Xt=t_X_i_k,
                mu_t=t_m_i_k,
            )
            T_X_k += transport.transform(X_k)
        T_X_k /= S_k

        # Compute the distribution of mu_{k+1}
        gamma_k = learning_rate(k)
        X_kp1 = (1 - gamma_k) * X_k + gamma_k * T_X_k

        # Compute Wasserstein distance
        diff = X_k - T_X_k
        w_dist = float(
            (gamma_k ** 2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))

        # Add to history
        if position_history:
            position_history.append(X_kp1)
        if distribution_history:
            distribution_history.append(
                D.DiscreteDistribution(support=X_kp1, weights=m)
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
    if distrib_sampler_history:
        to_return.append(distrib_sampler_history)

    return tuple(to_return)


# noinspection PyMissingOrEmptyDocstring,DuplicatedCode,PyUnboundLocalVariable
def compute_bwb_distribution_draw(
    distrib_sampler: DistributionSamplerP[D.DistributionDraw],
    learning_rate: t.Callable[[int], float],  # The \gamma_k schedule
    reg: float = 3e-3,  # Regularization of the convolutional method
    entrop_sharp=False,
    max_iter: int = 100_000,
    max_time: float = float("inf"),  # In seconds
    weights_history=False,
    distribution_history=False,
    distrib_sampler_history=False,
    report_every=10,
):
    # Warning for deprecation
    warnings.warn(
        "This function is deprecated. "
        "Use the DistributionDrawSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Paso 1: Sampling a mu_0
    mu_k: D.DistributionDraw = distrib_sampler.draw()
    dtype, device = mu_k.dtype, mu_k.device
    _log.info(f"dtype = {dtype}, device = {device}")

    gs_weights_k = mu_k.grayscale_weights

    # Create histories
    if weights_history:
        weights_history = [gs_weights_k]
    if distribution_history:
        distribution_history = [mu_k]
    if distrib_sampler_history:
        distrib_sampler_history = [[mu_k]]

    tic, toc = time.time(), time.time()
    diff_t = 0
    k = 0
    while (
        k < max_iter  # Reaches maximum iteration
        and toc - tic < max_time  # Reaches maximum time
    ):
        tic_ = time.time()
        if k % report_every == 0:
            _log.info(
                _bar
                + f" k = {k}, "
                + f"t = {toc - tic:.2f} [seg], "
                + f"Δt = {diff_t * 1000:.2f} [ms], "
                + f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} "
                  f"[ms/iter] "
                + _bar
            )

        m_k: D.DistributionDraw = distrib_sampler.draw()
        if distrib_sampler_history:
            distrib_sampler_history.append([m_k])

        # Compute the distribution of mu_{k+1}
        gamma_k = learning_rate(k)

        gs_weights_kp1, _ = bregman.convolutional_barycenter2d(
            A=[gs_weights_k, m_k.grayscale_weights],
            weights=[1 - gamma_k, gamma_k],
            reg=reg,
            entrop_sharp=entrop_sharp,
            numItermax=1_000,
            stopThr=1e-8,
            warn=False,
            log=True,
        )

        # Add to history
        if weights_history:
            weights_history.append(gs_weights_k)
        if distribution_history:
            mu_kp1 = D.DistributionDraw.from_grayscale_weights(
                gs_weights_kp1)
            distribution_history.append(mu_kp1)

        # Update
        k += 1
        gs_weights_k = gs_weights_kp1
        toc = time.time()
        diff_t = toc - tic_

    mu = D.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
    to_return = [mu]
    if weights_history:
        to_return.append(weights_history)
    if distribution_history:
        to_return.append(distribution_history)
    if distrib_sampler_history:
        to_return.append(distrib_sampler_history)

    return tuple(to_return)


# noinspection PyMissingOrEmptyDocstring,DuplicatedCode
def compute_bwb_distribution_draw_projected(
    distrib_sampler: DistributionSamplerP[D.DistributionDraw],
    projector,
    learning_rate: t.Callable[[int], float],  # The \gamma_k schedule
    reg: float = 3e-3,  # Regularization of the convolutional method
    entrop_sharp=False,
    max_iter: int = 100_000,
    max_time: float = float("inf"),  # In seconds
    weights_history=False,
    distribution_history=False,
    distrib_sampler_history=False,
    report_every=10,
):
    # Warning for deprecation
    warnings.warn(
        "This function is deprecated. "
        "Use the DistributionDrawSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Paso 1: Sampling a mu_0
    mu_k: D.DistributionDraw = distrib_sampler.draw()
    dtype, device = mu_k.dtype, mu_k.device
    _log.info(f"dtype = {dtype}, device = {device}")

    gs_weights_k = mu_k.grayscale_weights

    # Create histories
    if weights_history:
        weights_history = [gs_weights_k]
    if distribution_history:
        distribution_history = [mu_k]
    if distrib_sampler_history:
        distrib_sampler_history = [[mu_k]]

    tic, toc = time.time(), time.time()
    diff_t = 0
    k = 0
    while (
        k < max_iter  # Reaches maximum iteration
        and toc - tic < max_time  # Reaches maximum time
    ):
        tic_ = time.time()
        if k % report_every == 0:
            _log.info(
                _bar
                + f" k = {k}, "
                + f"t = {toc - tic:.2f} [seg], "
                + f"Δt = {diff_t * 1000:.2f} [ms], "
                + f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} "
                  f"[ms/iter] "
                + _bar
            )

        m_k: D.DistributionDraw = distrib_sampler.draw()
        if distrib_sampler_history:
            distrib_sampler_history.append([m_k])

        # Compute the distribution of mu_{k+1}
        gamma_k = learning_rate(k)
        _log.debug(f"gamma_k = {gamma_k:.6f}")

        gs_weights_kp1, _ = bregman.convolutional_barycenter2d(
            A=[gs_weights_k, m_k.grayscale_weights],
            weights=[1 - gamma_k, gamma_k],
            reg=reg,
            entrop_sharp=entrop_sharp,
            numItermax=1_000,
            stopThr=1e-8,
            warn=False,
            log=True,
        )

        # Project on Manifold
        gs_weights_kp1 = projector(gs_weights_kp1).to(gs_weights_k)

        # Add to history
        if weights_history:
            weights_history.append(gs_weights_k)
        if distribution_history:
            mu_kp1 = D.DistributionDraw.from_grayscale_weights(
                gs_weights_kp1)
            distribution_history.append(mu_kp1)

        # Update
        k += 1
        gs_weights_k = gs_weights_kp1
        toc = time.time()
        diff_t = toc - tic_

    # noinspection PyUnboundLocalVariable
    mu = D.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
    to_return = [mu]
    if weights_history:
        to_return.append(weights_history)
    if distribution_history:
        to_return.append(distribution_history)
    if distrib_sampler_history:
        to_return.append(distrib_sampler_history)

    return tuple(to_return)
