"""
Module for Stochastic Gradient Descent Algorithms in Wasserstein Space.
"""
import abc
import time
import typing as t
import warnings

import ot
import torch
from torch import linalg as LA

import bwb.distributions as dist
import bwb.distributions.utils as dist_utils
import bwb.logging_ as logging
import bwb.pot.bregman as bregman
import bwb.pot.transports as tpt
from wgan_gp.wgan_gp_vae.utils import ProjectorOnManifold
from .utils import (DetentionParameters, History, IterationParameters, Report,
                    ReportOptions, Schedule)
from ..protocols import HasDeviceDType

__all__ = [
    "Runnable",
    "BaseSGDW",
    "DiscreteDistributionSGDW",
    "DistributionDrawSGDW",
    "ConvDistributionDrawSGDW",
    "DebiesedDistributionDrawSGDW",
]

discrete_pos_wgt = tuple[torch.Tensor, torch.Tensor]

_log = logging.get_logger(__name__)
_bar = "=" * 5


class Runnable[DistributionT, pos_wgt_t](metaclass=abc.ABCMeta):
    """
    Interface for classes that can be run.
    """

    @abc.abstractmethod
    def run(
        self,
        pos_wgt_hist: bool = False,
        distr_hist: bool = False,
        pos_wgt_samp_hist: bool = False,
        distr_samp_hist: bool = False,
        include_dict: t.Optional[ReportOptions] = None
    ) -> DistributionT:
        """
        Run the algorithm.
        """
        ...


# MARK: BaseSGDW Class
# noinspection PyMethodOverriding
class BaseSGDW[DistributionT, pos_wgt_t](Runnable[DistributionT, pos_wgt_t],
                                         HasDeviceDType,
                                         metaclass=abc.ABCMeta):
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
    :param report_every: The frequency at which to report the metrics.
        Defaults to 10.
    :type report_every: int

    :raises TypeError: If ``learning_rate`` is not a callable or
        ``batch_size`` is not a callable or an integer.
    :raises ValueError: If ``learning_rate`` does not return a float or
        ``batch_size`` does not return an integer.
    """

    def __init__(
        self,
        distr_sampler: dist.DistributionSampler[DistributionT],
        step_scheduler,
        batch_size=1,
        tol: float = 1e-8,  # Tolerance to converge
        max_iter: int = 100_000,  # Maximum number of iterations
        max_time: float = float("inf"),  # Maximum time in seconds
        report_every: int = 10,  # Report every k iterations
    ):
        # Distribution sampler
        self.distr_sampler = distr_sampler
        """The distribution sampler for the algorithm."""

        # Schedule parameters
        self.schd = Schedule(step_scheduler, batch_size)

        # Detention parameters
        self.det_params = DetentionParameters(tol, max_iter, max_time)

        # Iteration metrics
        self.iter_params = IterationParameters(self.det_params)

        # History of the position and weights
        self.hist: History[DistributionT, pos_wgt_t] = History()

        # Report parameters
        self.report = Report(self.iter_params, report_every=report_every)

        # Values for dtype and device
        self.val = torch.tensor(1, dtype=self.dtype, device=self.device)
        """A tensor with value 1, used to pass to the device."""

        # Callback
        self._callback: t.Callable[[], None] = lambda: None

    @t.override
    @property
    def dtype(self) -> torch.dtype:
        return self.distr_sampler.dtype

    @t.override
    @property
    def device(self) -> torch.device:
        return self.distr_sampler.device

    @property
    def callback(self) -> t.Callable[[], None]:
        """
        Callback function to run at the end of each iteration.

        :return: The callback function.
        """
        return self._callback

    @callback.setter
    def callback(self, callback: t.Callable[[], None]) -> None:
        self._callback = callback

    @abc.abstractmethod
    def create_distribution(self, pos_wgt: pos_wgt_t) -> DistributionT:
        """
        Create a distribution from the position and weight.

        This method should create a distribution from the position and
        weight and return it.

        :param pos_wgt: The position and weight.
        :return: The distribution created from the position and weight.
        """
        pass

    @abc.abstractmethod
    def get_pos_wgt(self, mu: DistributionT) -> pos_wgt_t:
        """
        Get the position and weight from a distribution.

        This method should get the position and weight from a
        distribution and return it.

        :param mu: The distribution.
        :return: The position and weight from the distribution.
        """
        pass

    @abc.abstractmethod
    def first_sample(self) -> tuple[t.Sequence[DistributionT], pos_wgt_t]:
        """
        Draw the first sample from the distribution sampler. This
        corresponds to the first step of the algorithm.

        This method should draw the first sample from the distribution
        sampler and return it.

        :return: The first sample from the distribution sampler and the
            position and weight that come from the sample.
        """
        pass

    @abc.abstractmethod
    def update_pos_wgt(
        self,
        pos_wgt_k: pos_wgt_t,
        lst_mu_k: t.Sequence[DistributionT],
        gamma_k: float
    ) -> pos_wgt_t:
        """
        Update the position and weight for the next iteration.

        This method should update the position and weight for the next
        iteration and return it.

        :param pos_wgt_k: The position and weight that come from the
            current sample.
        :param lst_mu_k: The list of distributions drawn from the
            sampler at the current iteration.
        :param gamma_k: The learning rate for the next sample.
        """
        pass

    @t.final
    def set_schedules(
        self,
        step_schedule: t.Callable[[int], float],  # The \gamma_k schedule
        batch_size: t.Union[t.Callable[[int], int], int],  # The S_k schedule
    ):
        r"""
        Set the step and batch size schedules for the algorithm.

        :param step_schedule: A callable function that takes an integer
            argument :math:`k` and returns the step :math:`\gamma_k` for
            iteration :math:`k`.
        :param batch_size: A callable function that takes an integer
            argument :math:`k` and returns the batch size :math:`S_k` for
            iteration :math:`k`. Alternatively, it can be
            a constant integer value.
        :raises TypeError: If ``learning_rate`` is not a callable or
            ``batch_size`` is not a callable or an integer.
        :raises ValueError: If ``learning_rate`` does not return a
            float or ``batch_size`` does not return an integer.
        :return: The object itself.
        """
        self.schd.step_schedule = step_schedule
        self.schd.batch_size = batch_size
        return self

    @t.final
    def set_detention_params(
        self,
        tol: float = 1e-8,
        max_iter: int = 100_000,
        max_time: float = float("inf")
    ):
        """
        Set the detention parameters for the model.

        :param tol: The tolerance value for convergence.
            Defaults to 1e-8.
        :type tol: float
        :param max_iter: The maximum number of iterations.
            Defaults to 100_000.
        :type max_iter: int
        :param max_time: The maximum time allowed for the algorithm to
            run. Defaults to infinity.
        :type max_time: float
        :return: The object itself.
        :raises TypeError: If ``tol`` is not a real number, ``max_iter``
            is not an integer or ``max_time`` is not a real number.
        :raises ValueError: If ``tol`` is not positive, ``max_iter``
            is not positive or ``max_time`` is not positive.
        """
        self.det_params.tol = tol
        self.det_params.max_iter = max_iter
        self.det_params.max_time = max_time
        return self

    @t.final
    # noinspection PyPep8Naming
    def samp_distributions(self, S_k: int) -> t.Sequence[DistributionT]:
        """
        Draw S_k samples from the distribution sampler.

        This method should draw S_k samples from the distribution
        sampler and return them.

        :param S_k: The batch size for the current iteration.
        :return: A list of distributions drawn from the sampler.
        """
        return self.distr_sampler.sample(S_k)

    def _compute_wass_dist(
        self, pos_wgt_k: pos_wgt_t, pos_wgt_kp1: pos_wgt_t, gamma_k: float
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

    @t.final
    def compute_wass_dist(
        self, pos_wgt_k: pos_wgt_t, pos_wgt_kp1: pos_wgt_t, gamma_k: float
    ) -> float:
        """
        Compute the Wasserstein distance between two positions and weights.

        :param pos_wgt_k: The position and weight that come from the
            current sample.
        :param pos_wgt_kp1: The position and weight that come from the
            next sample.
        :param gamma_k: The learning rate for the next sample.
        """
        wass_dist = self._compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)
        return self.iter_params.update_wass_dist(wass_dist)

    def create_barycenter(self, pos_wgt: pos_wgt_t) -> DistributionT:
        """
        Create the barycenter from the position and weight. This method
        is called at the end of the algorithm. To use template pattern.
        """
        return self.create_distribution(pos_wgt)

    @t.final
    def init_algorithm(self) -> tuple[t.Sequence[DistributionT], pos_wgt_t]:
        """
        Initialize the algorithm.
        """
        # Step 1: Sampling a mu_0
        lst_mu_k, pos_wgt_k = self.first_sample()

        self.hist.init_histories(lst_mu_k, pos_wgt_k)

        return lst_mu_k, pos_wgt_k

    @t.final
    def step_algorithm(self, k: int, pos_wgt_k: pos_wgt_t) -> pos_wgt_t:
        """
        Run a step of the algorithm.
        """
        # Step 2: Draw S_k samples from the distribution sampler
        S_k = self.schd.batch_size(k)
        lst_mu_k = self.samp_distributions(S_k)

        # Step 3: Compute the distribution of mu_{k+1}
        gamma_k = self.schd.step_schedule(k)
        pos_wgt_kp1 = self.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

        # Step 4 (optional): Compute the Wasserstein distance
        self.compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)

        # Step 5 (optional): Add to history
        self.hist.update_histories(pos_wgt_kp1, lst_mu_k)

        # If the iteration is a multiple of report_every, print the report
        return pos_wgt_kp1

    @t.final
    def run(
        self,
        pos_wgt_hist=False,
        distr_hist=False,
        pos_wgt_samp_hist=False,
        distr_samp_hist=False,
        include_dict: t.Optional[ReportOptions] = None,
    ) -> DistributionT:
        """
        Run the algorithm.

        This method runs the algorithm and returns the final position
        weights and optional history data.

        :param pos_wgt_hist: Whether to include the position weights
            in the history.
        :param distr_hist: Whether to include the distributions in the
            history.
        :param pos_wgt_samp_hist: Whether to include the position
            weights of the samples in the history.
        :param distr_samp_hist: Whether to include the distributions
            of the samples in the history.
        :param include_dict: The options to include in the report.
        :return: A tuple containing the final position weights and
            optional history data.
        """
        self.hist.set_params(
            pos_wgt=pos_wgt_hist,
            distr=distr_hist,
            pos_wgt_samp=pos_wgt_samp_hist,
            distr_samp=distr_samp_hist,
            create_distribution=self.create_distribution,
            get_pos_wgt_from_dist=self.get_pos_wgt,
        )

        self.report.set_params(include_dict=include_dict)

        _, pos_wgt_k = self.init_algorithm()

        # The logic of the detention criteria and update are in the
        #   iterable class :class:`IterationParameters`
        for k in self.iter_params:
            # Run a step of the algorithm
            pos_wgt_k = self.step_algorithm(k, pos_wgt_k)

            # If the iteration is a multiple of report_every, print the report
            if self.report.is_report_iter():
                gamma_k = self.schd.step_schedule(k)
                _log.info(self.report.make_report(gamma_k))

            # Callback
            self.callback()

        barycenter = self.create_barycenter(pos_wgt_k)

        return barycenter


# MARK: SGDW with discrete distributions
class DiscreteDistributionSGDW(
    BaseSGDW[dist.DiscreteDistribution, discrete_pos_wgt]
):
    def __init__(
        self,
        transport: tpt.BaseTransport,
        distr_sampler: dist.DistributionSampler[dist.DiscreteDistribution],
        step_scheduler: t.Callable[[int], float],
        batch_size: t.Union[t.Callable[[int], int], int] = 1,
        alpha: float = 1.0,
        tol: float = 1e-8,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
        report_every=10,
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            step_scheduler=step_scheduler,
            batch_size=batch_size,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
            report_every=report_every,
        )
        self.transport = transport
        self.alpha = alpha
        self.include_w_dist = True

    @t.final
    @t.override
    def create_distribution(
        self, pos_wgt: discrete_pos_wgt
    ) -> dist.DiscreteDistribution:
        X_k, m = pos_wgt
        return dist.DiscreteDistribution(support=X_k, weights=m)

    @t.final
    @t.override
    def get_pos_wgt(self, mu: dist.DiscreteDistribution) -> discrete_pos_wgt:
        return mu.enumerate_nz_support_(), mu.nz_probs

    @t.final
    @t.override
    def first_sample(
        self,
    ) -> tuple[t.Sequence[dist.DiscreteDistribution], discrete_pos_wgt]:
        mu_0: dist.DiscreteDistribution = self.distr_sampler.draw()
        X_k, m = dist_utils.partition(
            X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=self.alpha
        )
        X_k, m = X_k.to(self.val), m.to(self.val)
        return [mu_0], (X_k, m)

    @t.final
    @t.override
    def update_pos_wgt(
        self,
        pos_wgt_k: discrete_pos_wgt,
        lst_mu_k: t.Sequence[dist.DiscreteDistribution],
        gamma_k: float,
    ) -> discrete_pos_wgt:
        X_k, m = pos_wgt_k
        T_X_k: torch.Tensor = torch.zeros_like(
            X_k, dtype=self.dtype, device=self.device
        )
        S_k = len(lst_mu_k)
        for mu_i_k in lst_mu_k:
            X_i_k, m_i_k = self.get_pos_wgt(mu_i_k)
            X_i_k, m_i_k = X_i_k.to(self.val), m_i_k.to(self.val)
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
        pos_wgt_k: discrete_pos_wgt,
        pos_wgt_kp1: discrete_pos_wgt,
        gamma_k: float
    ):
        X_k, m = pos_wgt_k
        X_kp1, m = pos_wgt_kp1
        diff = X_k - X_kp1
        w_dist = float(
            (gamma_k ** 2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        return w_dist


# IDEA: Divide the class for the projected version
# MARK: SGDW with distributions based in draws
class DistributionDrawSGDW(
    BaseSGDW[dist.DistributionDraw, torch.Tensor], metaclass=abc.ABCMeta
):
    def __init__(
        self,
        distr_sampler: dist.DistributionSampler[dist.DistributionDraw],
        step_scheduler: t.Callable[[int], float],
        batch_size: t.Union[t.Callable[[int], int], int] = 1,
        projector: t.Optional[ProjectorOnManifold] = None,
        proj_every: int = 1,
        tol: float = 0,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
        report_every=10,
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            step_scheduler=step_scheduler,
            batch_size=batch_size,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
            report_every=report_every,
        )
        self.conv_bar_kwargs = None
        self.projector = projector
        self.proj_every = proj_every

        self.set_geodesic_params()

    @t.final
    @t.override
    def create_distribution(
        self,
        pos_wgt: torch.Tensor
    ) -> dist.DistributionDraw:
        gs_weights_k = pos_wgt
        return dist.DistributionDraw.from_grayscale_weights(gs_weights_k)

    @t.final
    @t.override
    def get_pos_wgt(self, mu: dist.DistributionDraw) -> torch.Tensor:
        return mu.grayscale_weights

    @t.final
    @t.override
    def first_sample(
        self,
    ) -> tuple[t.Sequence[dist.DistributionDraw], torch.Tensor]:
        mu_0: dist.DistributionDraw = self.distr_sampler.draw()
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
        self.conv_bar_kwargs = dict(
            reg=reg,
            method=method,
            numItermax=num_iter_max,
            stopThr=stop_thr,
            verbose=verbose,
            log=False,
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
        lst_mu_k: t.Sequence[dist.DistributionDraw],
        gamma_k: float
    ) -> torch.Tensor:
        # TODO: REFACTOR THIS!
        S_k = len(lst_mu_k)
        gs_weights_kp1 = self._compute_geodesic(
            [pos_wgt_k] + [mu_k.grayscale_weights for mu_k in lst_mu_k],
            [1 - gamma_k] + [gamma_k / S_k] * S_k
        )
        if (
            self.projector is not None
            and self.proj_every is not None
            and self.proj_every > 0
            and self.iter_params.k % self.proj_every == self.proj_every - 1
        ):
            gs_weights_kp1 = self.projector(gs_weights_kp1).to(self.val)
        return gs_weights_kp1

    @t.final
    @t.override
    def create_barycenter(
        self,
        pos_wgt: torch.Tensor
    ) -> dist.DistributionDraw:
        # In the last iteration, project to the manifold
        if self.projector is not None:
            pos_wgt = self.projector(pos_wgt).to(self.val)
        bar = self.create_distribution(pos_wgt)
        return bar


class ConvDistributionDrawSGDW(DistributionDrawSGDW):
    @t.final
    @t.override
    def _compute_geodesic(self, gs_weights_lst_k, lst_gamma_k) -> torch.Tensor:
        return bregman.convolutional_barycenter2d(
            A=torch.stack(gs_weights_lst_k),
            weights=torch.as_tensor(lst_gamma_k, dtype=self.dtype,
                                    device=self.device),
            **self.conv_bar_kwargs,
        )


class DebiesedDistributionDrawSGDW(DistributionDrawSGDW):
    @t.final
    @t.override
    def _compute_geodesic(self, gs_weights_lst_k, lst_gamma_k) -> torch.Tensor:
        return ot.bregman.convolutional_barycenter2d_debiased(
            A=torch.stack(gs_weights_lst_k),
            weights=torch.as_tensor(lst_gamma_k, dtype=self.dtype,
                                    device=self.device),
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
    distrib_sampler: dist.DistributionSampler[dist.DiscreteDistribution],
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
    mu_0: dist.DiscreteDistribution = distrib_sampler.draw()
    dtype, device = mu_0.dtype, mu_0.device

    # Compute locations through the partition
    _log.info("Computing initial weights")
    tic = time.time()
    X_k, m = dist_utils.partition(
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
            dist.DiscreteDistribution(support=X_k, weights=m)]
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
            t_mu_i_k: dist.DiscreteDistribution = distrib_sampler.draw()
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
                dist.DiscreteDistribution(support=X_kp1, weights=m)
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
    distrib_sampler: dist.DistributionSampler[dist.DistributionDraw],
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
    mu_k: dist.DistributionDraw = distrib_sampler.draw()
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

        m_k: dist.DistributionDraw = distrib_sampler.draw()
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
            mu_kp1 = dist.DistributionDraw.from_grayscale_weights(
                gs_weights_kp1)
            distribution_history.append(mu_kp1)

        # Update
        k += 1
        gs_weights_k = gs_weights_kp1
        toc = time.time()
        diff_t = toc - tic_

    mu = dist.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
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
    distrib_sampler: dist.DistributionSampler[dist.DistributionDraw],
    projector: ProjectorOnManifold,
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
    mu_k: dist.DistributionDraw = distrib_sampler.draw()
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

        m_k: dist.DistributionDraw = distrib_sampler.draw()
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
            mu_kp1 = dist.DistributionDraw.from_grayscale_weights(
                gs_weights_kp1)
            distribution_history.append(mu_kp1)

        # Update
        k += 1
        gs_weights_k = gs_weights_kp1
        toc = time.time()
        diff_t = toc - tic_

    # noinspection PyUnboundLocalVariable
    mu = dist.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
    to_return = [mu]
    if weights_history:
        to_return.append(weights_history)
    if distribution_history:
        to_return.append(distribution_history)
    if distrib_sampler_history:
        to_return.append(distrib_sampler_history)

    return tuple(to_return)
