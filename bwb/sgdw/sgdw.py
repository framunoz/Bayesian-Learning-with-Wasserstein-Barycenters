"""
Module for Stochastic Gradient Descent Algorithms in Wasserstein Space.
"""

import abc
from copy import deepcopy
from functools import partial
from typing import (
    Callable,
    Literal,
    Protocol,
    final,
    override,
)
from typing import (
    Sequence as Seq,
)

import ot
import torch
from torch import linalg as LA

import bwb.distributions as D
import bwb.distributions.utils as D_utils
import bwb.logging_ as logging
from bwb import protocols as P
from bwb.sgdw import utils

__all__ = [
    "SGDW",
    "BaseSGDW",
    "CallbackFn",
    "DiscreteDistributionSGDW",
    "DistributionDrawSGDW",
    "Runnable",
    "_convolutional_methods",
]

type DiscretePosWgt = tuple[torch.Tensor, torch.Tensor]
type DistDrawPosWgt = torch.Tensor
type CallbackFn = Callable[[dict], None]
type ConvolutionalFn = Callable[..., torch.Tensor]
type ConvolutionalArg = Literal["conv", "debiased"] | ConvolutionalFn

_log = logging.get_logger(__name__)
_bar = "=" * 5

_convolutional_methods = {
    "conv": partial(
        ot.bregman.convolutional_barycenter2d,
        reg=3e-3,
        method="sinkhorn",
        numItermax=10_000,
        stopThr=1e-4,
        verbose=False,
        warn=False,
    ),
    "debiased": partial(
        ot.bregman.convolutional_barycenter2d_debiased,
        reg=1e-2,
        method="sinkhorn",
        numItermax=10_000,
        stopThr=1e-3,
        verbose=False,
        warn=False,
    ),
}


def _get_conv_function(conv_bar_strategy: ConvolutionalArg) -> ConvolutionalFn:
    """
    Get the convolutional function from the convolutional strategy.
    """
    if isinstance(conv_bar_strategy, str):
        if conv_bar_strategy not in _convolutional_methods:
            raise ValueError(f"Unknown convolutional strategy '{conv_bar_strategy}'.")
        return _convolutional_methods[conv_bar_strategy]
    elif callable(conv_bar_strategy):
        return conv_bar_strategy
    raise TypeError("The convolutional strategy must be a string or a callable.")


def _transport_source(xt: torch.Tensor, plan: torch.Tensor):
    r"""
    Transport the source to the target. This is done by computing the
    barycentric mapping and then computing the transported source.
    Code from: `ot.da.BaseTransport.transform
    <https://pythonot.github.io/_modules/ot/da.html#BaseTransport.transform>`_.

    This function use the following equation:
    .. math::
        T_{\pi}(x) = \frac{\int y \pi_x(dy)}{\int \pi_x(dy)}

    where :math:`\pi_x(dy) = \int \pi(dx, dy)`

    :param xt: target distribution.
    :param plan: transport plan.
    :return: transported source.
    """
    nx = ot.backend.TorchBackend()
    # perform standard barycentric mapping
    transp = plan / nx.sum(plan, axis=1)[:, None]
    # set nan values to 0
    transp = nx.nan_to_num(transp, nan=0, posinf=0, neginf=0)
    # compute transported source
    return nx.dot(transp, xt)


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


class DistributionSamplerP[DistributionT](P.HasDeviceDTypeP, Protocol):
    """
    Protocol for distribution samplers.
    """

    def sample(self, n: int) -> Seq[DistributionT]:
        """
        Sample a sequence of distributions.

        :param n: The number of distributions to sample.
        :return: The sequence of distributions.
        """


# TODO: Hacer que el callback acepte como argumento información
#  relevante de la iteración. Por ejemplo, para el sgdw es relevante
#  la lista de distribuciones muestreadas y la posición y peso de la
#  última iteración. Lo que se podría hacer, es crear un diccionario en
#  donde se guarden los valores relevantes de la iteración, se retorne
#  este diccionario a las funciones init_algorithm y step_algorithm,
#  y se pase este diccionario al callback. De esta forma, el callback
#  tendrá toda la información relevante de la iteración. Esto podría
#  reducir significativamente la cantidad de código desarrollado,
#  además de dejarlo mucho más limpio,
#  quitando a la mayoría de loggers que hay en el módulo wrappers,
#  además de simplificar los wrappers del módulo plotters.
# TODO: Se podría simplificar las funciones init_algorithm y
#  first_sample. El método first_sample se podría eliminar y en su
#  lugar se podría hacer que init_algorithm devuelva un diccionario
class SGDW[DistributionT, PosWgtT](
    Runnable[DistributionT], P.HasDeviceDType, metaclass=abc.ABCMeta
):
    r"""
    Base class that works as interface for the Stochastic Gradient Descent
    in Wasserstein Space classes.
    """

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[DistributionT],
        schd: utils.Schedule,
        det_params: utils.DetentionParameters,
        iter_params: utils.IterationParameters,
        callback: CallbackFn,
        dict_log: dict,
    ):
        self.distr_sampler = distr_sampler
        self.schd = schd
        self.det_params = det_params
        self.iter_params = iter_params

        self.callback: CallbackFn = callback
        self.dict_log: dict = dict_log

    @final
    @override
    def run(self) -> DistributionT:
        """
        Run the algorithm.
        """
        _, pos_wgt_k = self.init_algorithm()

        # The logic of the detention criteria and update are in the
        #   iterable class :class:`IterationParameters`
        for k in self.iter_params:
            # Run a step of the algorithm
            _, pos_wgt_kp1 = self.step_algorithm(k, pos_wgt_k)

            # Step 4 (optional): Compute the Wasserstein distance
            if self.iter_params.is_wass_dist_iter():
                gamma_k = self.schd.step_schedule(k)
                wass_dist = self._compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)
                self.update_wass_dist(wass_dist)

            # Update the position and weight
            pos_wgt_k = pos_wgt_kp1

            # Callback to do extra instructions at the end of each iteration
            self.callback(self.dict_log)
            # Clear the dictionary log
            self.dict_log.clear()

        barycenter = self.create_barycenter(pos_wgt_k)

        return barycenter

    def init_algorithm(self) -> tuple[Seq[DistributionT], PosWgtT]:
        """
        Initialize the algorithm.
        """
        # Step 1: Sampling a mu_0
        lst_mu_0, pos_wgt_0 = self.first_sample()

        self.dict_log["lst_mu"] = lst_mu_0
        self.dict_log["pos_wgt"] = pos_wgt_0

        return lst_mu_0, pos_wgt_0

    def step_algorithm(
        self, k: int, pos_wgt_k: PosWgtT
    ) -> tuple[Seq[DistributionT], PosWgtT]:
        """
        Run a step of the algorithm.
        """
        # Step 2: Draw S_k samples from the distribution sampler
        S_k = self.schd.batch_size(k)
        lst_mu_k = self.distr_sampler.sample(S_k)

        # Step 3: Compute the distribution of mu_{k+1}
        gamma_k = self.schd.step_schedule(k)
        pos_wgt_kp1 = self.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

        self.dict_log["lst_mu"] = lst_mu_k
        self.dict_log["pos_wgt"] = pos_wgt_kp1
        self.dict_log["k"] = k
        self.dict_log["gamma_k"] = gamma_k

        return lst_mu_k, pos_wgt_kp1

    def update_wass_dist(self, wass_dist: float) -> float:
        """
        Update the Wasserstein distance.
        """
        wass_dist_smooth = self.iter_params.update_wass_dist(wass_dist)

        self.dict_log["wass_dist"] = wass_dist
        self.dict_log["wass_dist_smooth"] = wass_dist_smooth

        return wass_dist_smooth

    def _additional_repr_(self, sep: str, tab: str, n_tab: int, new_line: str) -> str:
        """
        Additional representation for the class.
        """
        return ""

    def _repr_(self, sep: str, tab: str, n_tab: int, new_line: str) -> str:
        """
        Representation for the class.
        """
        to_return = tab * n_tab + "- " + self.__class__.__name__ + ":"
        add_repr = self._additional_repr_(sep, tab, n_tab + 1, new_line)
        if add_repr:
            to_return += new_line + add_repr
        if to_return.endswith(sep):
            to_return = to_return[: -len(sep)]

        return to_return

    def __repr__(self) -> str:
        new_line = "\n"
        tab = "    "
        sep = new_line

        return self._repr_(sep, tab, 0, new_line)

    @final
    @override
    @property
    def dtype(self) -> torch.dtype:
        return self.distr_sampler.dtype

    @final
    @override
    @property
    def device(self) -> torch.device:
        return self.distr_sampler.device

    @abc.abstractmethod
    def first_sample(self) -> tuple[Seq[DistributionT], PosWgtT]:
        """
        Draw the first sample from the distribution sampler.
        """
        ...

    @abc.abstractmethod
    def update_pos_wgt(
        self,
        pos_wgt_k: PosWgtT,
        lst_mu_k: Seq[DistributionT],
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

    @abc.abstractmethod
    def get_pos_wgt(self, mu: DistributionT) -> PosWgtT:
        """
        Get a representation of the position and weight from a distribution.
        """
        ...

    @abc.abstractmethod
    def as_distribution(self, pos_wgt: PosWgtT) -> DistributionT:
        """
        Create a distribution from the position and weight.
        """
        ...

    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        """
        Create the barycenter from the position and weight.
        """
        return self.as_distribution(pos_wgt)


# MARK: BaseSGDW Class
# noinspection PyMethodOverriding
class BaseSGDW[DistributionT, PosWgtT](
    SGDW[DistributionT, PosWgtT], metaclass=abc.ABCMeta
):
    r"""
    Base class for Stochastic Gradient Descent in Wasserstein Space.

    This class provides a base implementation for Stochastic Gradient
    Descent in Wasserstein Space. It defines the common attributes and
    methods used by the derived classes.

    :param step_scheduler: A callable function that takes an integer
        argument :math:`k` and returns the learning rate :math:`\gamma_k`
        for iteration :math:`k`. Alternatively, it can be a constant
        float value.
    :param batch_size: A callable function that takes an integer
        argument :math:`k` and returns the batch size :math:`S_k` for
        iteration :math:`k`. Alternatively, it can be a constant integer
        value.
    :param tol: The tolerance value for convergence. Defaults to 1e-8.
    :param max_iter: The maximum number of iterations.
        Defaults to 100_000.
    :param max_time: The maximum time allowed for the algorithm to run.
        Defaults to infinity.
    :param wass_dist_every: The number of iterations to compute the
        Wasserstein distance. Defaults to 1.

    :raises TypeError: If ``learning_rate`` is not a callable or
        ``batch_size`` is not a callable or an integer.
    :raises ValueError: If ``learning_rate`` does not return a float or
        ``batch_size`` does not return an integer.
    """

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[DistributionT],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg,
        tol: float,
        min_iter: int,
        max_iter: int,
        max_time: float,
        wass_dist_every: int,
        callback: CallbackFn,
    ):
        schd = utils.Schedule(step_scheduler, batch_size)
        det_params = utils.DetentionParameters(
            tol, min_iter, max_iter, max_time, wass_dist_every
        )
        iter_params = utils.IterationParameters(det_params, length_ema=5)

        super().__init__(
            distr_sampler,
            schd,
            det_params,
            iter_params,
            callback,
            {},
        )

        # A value to pass to the device and dtype
        self._val = torch.tensor(1, dtype=self.dtype, device=self.device)

    @override
    def _additional_repr_(self, sep: str, tab: str, n_tab: int, new_line: str) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + "distr_sampler=" + repr(self.distr_sampler) + sep
        to_return += space + "iter_params=" + repr(self.iter_params) + sep
        to_return += space + "det_params=" + repr(self.det_params) + sep

        return to_return


# MARK: SGDW with distributions based in draws
@final
class DistributionDrawSGDW(
    BaseSGDW[D.DistributionDraw, DistDrawPosWgt],
):
    r"""
    Class for Stochastic Gradient Descent in Wasserstein Space with
    distributions based in draws.

    :param distr_sampler: The distribution sampler. It must be an
        instance of :class:`DistributionSamplerP`.
    :param step_scheduler: The step scheduler. It must be a callable
        function that takes an integer argument :math:`k` and returns
        the learning rate :math:`\gamma_k` for iteration :math:`k`.
        Alternatively, it can be a constant float value.
    :param conv_bar_strategy: The convolutional strategy to compute the
        barycenter. It can be a string with the values ``"conv"`` or
        ``"debiased"``. Alternatively, it can be a callable function.
        If it is a callable function, it must have the signature
        ``(A: torch.Tensor, weights: torch.Tensor, **conv_bar_kwargs)
        -> torch.Tensor.``
    :param conv_bar_kwargs: Additional keyword arguments for the
        convolutional strategy. Defaults is an empty dictionary. For
        further information, see `ot.bregman.convolution_barycenter2d
        <https://pythonot.github.io/gen_modules/ot.bregman.html#ot.bregman.convolutional_barycenter2d>`_.
    :param batch_size: The batch size. It must be a callable function
        that takes an integer argument :math:`k` and returns the batch
        size :math:`S_k` for iteration :math:`k`. Alternatively, it can
        be a constant integer value. Defaults to 1.
    :param tol: The tolerance value for convergence. Defaults to 0.
    :param max_iter: The maximum number of iterations. Defaults to 1_000.
    :param max_time: The maximum time allowed for the algorithm to run.
        Defaults to infinity.
    :param wass_dist_every: The number of iterations to compute the
        Wasserstein distance. Defaults to 10.
    :param wass_dist_kwargs: Additional keyword arguments for the
        function ``ot.solve_sample``. Defaults is an empty dictionary.
        For further information, see `ot.solve_sample
        <https://pythonot.github.io/all.html#ot.solve_sample>`_.
    """

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[D.DistributionDraw],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg = 1,
        conv_bar_strategy: ConvolutionalArg = "debiased",
        conv_bar_kwargs: dict | None = None,
        tol: float = 0,
        min_iter: int = 0,
        max_iter: int = 1_000,
        max_time: float = float("inf"),
        wass_dist_every: int = 10,
        callback: CallbackFn = lambda x: None,
        wass_dist_kwargs: dict | None = None,
    ):
        super().__init__(
            distr_sampler,
            step_scheduler,
            batch_size,
            tol,
            min_iter,
            max_iter,
            max_time,
            wass_dist_every,
            callback,
        )
        self.conv_bar_strategy: ConvolutionalFn = _get_conv_function(conv_bar_strategy)
        self.conv_bar_kwargs: dict = (
            deepcopy(conv_bar_kwargs) if conv_bar_kwargs is not None else {}
        )
        self.wass_dist_kwargs: dict = (
            wass_dist_kwargs if wass_dist_kwargs is not None else {}
        )

    @override
    def as_distribution(self, pos_wgt: DistDrawPosWgt) -> D.DistributionDraw:
        return D.DistributionDraw.from_grayscale_weights(pos_wgt)

    @override
    def get_pos_wgt(self, mu: D.DistributionDraw) -> DistDrawPosWgt:
        return mu.grayscale_weights

    @override
    def first_sample(self) -> tuple[Seq[D.DistributionDraw], DistDrawPosWgt]:
        mu_0: D.DistributionDraw = self.distr_sampler.sample(1)[0]
        return [mu_0], mu_0.grayscale_weights

    @override
    def update_pos_wgt(
        self,
        pos_wgt_k: DistDrawPosWgt,
        lst_mu_k: Seq[D.DistributionDraw],
        gamma_k: float,
    ) -> DistDrawPosWgt:
        S_k = len(lst_mu_k)
        gs_weights_lst_k = (
            [pos_wgt_k] + [mu_k.grayscale_weights for mu_k in lst_mu_k]
        )  # fmt: skip
        lst_gamma_g = [1 - gamma_k] + [gamma_k / S_k] * S_k
        gs_weights_kp1 = self.conv_bar_strategy(
            A=torch.stack(gs_weights_lst_k),
            weights=torch.as_tensor(lst_gamma_g, dtype=self.dtype, device=self.device),
            **self.conv_bar_kwargs,
        )
        return gs_weights_kp1

    @override
    @logging.register_total_time_method(_log)
    def _compute_wass_dist(
        self,
        pos_wgt_k: DistDrawPosWgt,
        pos_wgt_kp1: DistDrawPosWgt,
        gamma_k: float,
    ) -> float:
        return D.wass_distance(
            self.as_distribution(pos_wgt_k),
            self.as_distribution(pos_wgt_kp1),
            **self.wass_dist_kwargs,
        )


# MARK: SGDW with discrete distributions
@final
class DiscreteDistributionSGDW(BaseSGDW[D.DiscreteDistribution, DiscretePosWgt]):
    r"""
    Class for Stochastic Gradient Descent in Wasserstein Space with
    discrete distributions.

    :param distr_sampler: The distribution sampler. It must be an
        instance of :class:`DistributionSamplerP`.
    :param step_scheduler: The step scheduler. It must be a callable
        function that takes an integer argument :math:`k` and returns
        the learning rate :math:`\gamma_k` for iteration :math:`k`.
        Alternatively, it can be a constant float value.
    :param batch_size: The batch size. It must be a callable function
        that takes an integer argument :math:`k` and returns the batch
        size :math:`S_k` for iteration :math:`k`. Alternatively, it can
        be a constant integer value. Defaults to 1.
    :param alpha: The value of the partition parameter. Defaults to 1.
    :param tol: The tolerance value for convergence. Defaults to 1e-8.
    :param max_iter: The maximum number of iterations. Defaults to
        1_000.
    :param max_time: The maximum time allowed for the algorithm to run.
        Defaults to infinity.
    :param wass_dist_every: The number of iterations to compute the
        Wasserstein distance. Defaults to 1.
    :param solve_sample_kwargs: Additional keyword arguments for the
        function ``ot.solve_sample``. Defaults is an empty dictionary.
        For further information, see `ot.solve_sample
        <https://pythonot.github.io/all.html#ot.solve_sample>`_.
    """

    def __init__(
        self,
        distr_sampler: DistributionSamplerP[D.DiscreteDistribution],
        step_scheduler: utils.StepSchedulerArg,
        batch_size: utils.BatchSizeArg = 1,
        alpha: float = 1.0,
        tol: float = 1e-3,
        min_iter: int = 0,
        max_iter: int = 1_000,
        max_time: float = float("inf"),
        wass_dist_every: int = 1,
        callback: CallbackFn = lambda x: None,
        solve_sample_kwargs: dict | None = None,
    ):
        super().__init__(
            distr_sampler,
            step_scheduler,
            batch_size,
            tol,
            min_iter,
            max_iter,
            max_time,
            wass_dist_every,
            callback,
        )
        self.alpha = alpha
        self.include_w_dist = True
        self.solve_sample_kwargs: dict = (
            solve_sample_kwargs if solve_sample_kwargs is not None else {}
        )

    @override
    def as_distribution(self, pos_wgt: DiscretePosWgt) -> D.DiscreteDistribution:
        X_k, m = pos_wgt
        return D.DiscreteDistribution(support=X_k, weights=m)

    @override
    def get_pos_wgt(self, mu: D.DiscreteDistribution) -> DiscretePosWgt:
        return (
            mu.enumerate_nz_support_().to(self._val),
            mu.nz_probs.to(self._val),
        )

    @override
    def first_sample(
        self,
    ) -> tuple[Seq[D.DiscreteDistribution], DiscretePosWgt]:
        mu_0: D.DiscreteDistribution = self.distr_sampler.sample(1)[0]
        X_k, m = D_utils.partition(
            X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=self.alpha
        )
        X_k, m = X_k.to(self._val), m.to(self._val)
        return [mu_0], (X_k, m)

    @override
    def update_pos_wgt(
        self,
        pos_wgt_k: DiscretePosWgt,
        lst_mu_k: Seq[D.DiscreteDistribution],
        gamma_k: float,
    ) -> DiscretePosWgt:
        X_k, m = pos_wgt_k
        T_X_k: torch.Tensor = torch.zeros_like(
            X_k, dtype=self.dtype, device=self.device
        )
        S_k = len(lst_mu_k)
        for mu_i_k in lst_mu_k:
            X_i_k, m_i_k = self.get_pos_wgt(mu_i_k)
            m_i_k /= torch.sum(m_i_k)
            res: ot.utils.OTResult = ot.solve_sample(
                X_k, X_i_k, m, m_i_k, **self.solve_sample_kwargs
            )
            T_X_k += _transport_source(X_i_k, res.plan)
        T_X_k /= S_k
        # noinspection PyTypeChecker
        X_kp1: torch.Tensor = (1 - gamma_k) * X_k + gamma_k * T_X_k
        return X_kp1, m

    @override
    def _compute_wass_dist(
        self,
        pos_wgt_k: DiscretePosWgt,
        pos_wgt_kp1: DiscretePosWgt,
        gamma_k: float,
    ):
        X_k, _ = pos_wgt_k
        X_kp1, m = pos_wgt_kp1
        diff = X_k - X_kp1
        w_dist = float((gamma_k**2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        return w_dist
