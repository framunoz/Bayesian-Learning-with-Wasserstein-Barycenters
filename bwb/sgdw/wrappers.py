"""
This module contains the wrappers for the SGDW algorithm.
"""
# TODO: ANNOTATE THE METHODS
import abc
from copy import deepcopy
from datetime import timedelta
from typing import (Callable, override, Sequence as Seq, TypedDict)

import bwb.logging_ as logging
from bwb.sgdw.sgdw import CallbackFn, SGDW

__all__ = [
    "ProjectorFn",
    "ReportProxy",
    "LogPosWgtIterProxy",
    "LogDistrIterProxy",
    "LogPosWgtSampledProxy",
    "LogDistrSampledProxy",
    "SGDWProjectedDecorator",
]

_log = logging.get_logger(__name__)

type ProjectorFn[PosWgtT] = Callable[[PosWgtT], PosWgtT]


class SGDWBaseWrapper[DistributionT, PosWgtT](SGDW[DistributionT, PosWgtT]):
    """
    Base class for the wrappers of the SGDW algorithm.
    """

    def __init__(self, wrapee: SGDW[DistributionT, PosWgtT]):
        super().__init__(
            wrapee.distr_sampler,
            wrapee.schd,
            wrapee.det_params,
            wrapee.iter_params
        )
        self.wrapee = wrapee

    def _repr_(self, sep: str, tab: str, n_tab: int, new_line: str) -> str:
        space = tab * n_tab
        to_return = space + "- " + self.__class__.__name__ + ":"
        add_repr = self._additional_repr_(sep, tab, n_tab + 1, new_line)
        if add_repr:
            to_return += new_line + add_repr
        wrapee_repr = self.wrapee._repr_(sep, tab, n_tab, new_line)
        to_return += new_line + wrapee_repr
        return to_return

    @override
    @property
    def callback(self) -> CallbackFn:
        return self.wrapee.callback

    @callback.setter
    def callback(self, value: CallbackFn) -> None:
        self.wrapee.callback = value

    @override
    def first_sample(self) -> tuple[Seq[DistributionT], PosWgtT]:
        return self.wrapee.first_sample()

    @override
    def init_algorithm(self) -> tuple[Seq[DistributionT], PosWgtT]:
        return self.wrapee.init_algorithm()

    @override
    def update_pos_wgt(
        self,
        pos_wgt_k: PosWgtT,
        lst_mu_k: Seq[DistributionT],
        gamma_k: float,
    ) -> PosWgtT:
        return self.wrapee.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

    @override
    def _compute_wass_dist(
        self, pos_wgt_k: PosWgtT, pos_wgt_kp1: PosWgtT, gamma_k: float
    ) -> float:
        return self.wrapee._compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)

    @override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[Seq[DistributionT], PosWgtT]:
        return self.wrapee.step_algorithm(k, pos_wgt_k)

    @override
    def get_pos_wgt(self, mu: DistributionT) -> PosWgtT:
        return self.wrapee.get_pos_wgt(mu)

    @override
    def create_distribution(self, pos_wgt: PosWgtT) -> DistributionT:
        return self.wrapee.create_distribution(pos_wgt)

    @override
    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        return self.wrapee.create_barycenter(pos_wgt)


class ReportOptions(TypedDict, total=False):
    """
    This class contains the report options for the algorithm.
    """
    iter: bool
    w_dist: bool
    step_schd: bool
    total_time: bool
    dt: bool
    dt_per_iter: bool


class ReportProxy[DistributionT, PosWgtT](
    SGDWBaseWrapper[DistributionT, PosWgtT]
):
    """
    Proxy class for the SGDW algorithm to report relevant iteration
    information.
    """
    INCLUDE_OPTIONS = ReportOptions(
        iter=True,
        w_dist=False,
        step_schd=True,
        total_time=True,
        dt=False,
        dt_per_iter=True,
    )
    include_dict: ReportOptions

    def __init__(
        self,
        wrapee: SGDW[DistributionT, PosWgtT], *,
        report_every: int = 10,
        len_bar: int = 5,
        include_dict: ReportOptions = None,
        log=_log,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(wrapee)
        self.report_every = report_every
        self.len_bar = len_bar
        self.include_dict = deepcopy(self.INCLUDE_OPTIONS)
        if include_dict is not None:
            self.include_dict.update(include_dict)
        self.log = log
        self.level = level

    @override
    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"report_every={self.report_every}" + sep
        level = {
            logging.DEBUG:    "DEBUG",
            logging.INFO:     "INFO",
            logging.WARNING:  "WARNING",
            logging.ERROR:    "ERROR",
            logging.CRITICAL: "CRITICAL",
        }
        to_return += space + f"level={level.get(self.level, self.level)}" + sep
        log_repr = f"'{self.log.name}': {level[self.log.level]}"
        to_return += (space + f"log={log_repr}" + sep)
        return to_return

    def make_report(self) -> str:
        """
        Generate the report for the algorithm.
        """

        bar = "=" * self.len_bar

        report = bar + " "

        k = self.iter_params.k
        gamma_k = self.schd.step_schedule(k)

        if self.include_dict["iter"]:
            report += f"k = {k}, "

        if self.include_dict["w_dist"]:
            report += f"Wass. dist. = {self.iter_params.w_dist:.6f}, "

        if self.include_dict["step_schd"]:
            report += f"step = {gamma_k:.2%}, "

        if self.include_dict["total_time"]:
            total_time = self.iter_params.total_time
            time_fmt = str(timedelta(seconds=total_time))[:-4]
            report += f"total time = {time_fmt}, "

        if self.include_dict["dt"]:
            report += f"Δt = {self.iter_params.diff_t * 1000:.2f} [ms], "

        if self.include_dict["dt_per_iter"]:
            dt_per_iter = (self.iter_params.total_time * 1000
                           / (self.iter_params.k + 1))
            report += f"Δt per iter. = {dt_per_iter:.2f} [ms/iter], "

        if report.endswith(", "):
            report = report[:-2]

        report += " " + bar

        return report

    def is_report_iter(self) -> bool:
        """
        Check if the current iteration should be reported.
        """
        return self.iter_params.k % self.report_every == 0

    @override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[Seq[DistributionT], PosWgtT]:
        result = super().step_algorithm(k, pos_wgt_k)

        if self.is_report_iter():
            self.log.log(self.level, self.make_report())

        return result


class BaseLogProxy[DistributionT, PosWgtT, RegisterValueT](
    SGDWBaseWrapper[DistributionT, PosWgtT],
    metaclass=abc.ABCMeta,
):

    def __init__(self, wrapee: SGDW[DistributionT, PosWgtT]) -> None:
        super().__init__(wrapee)
        self.register_lst: list[RegisterValueT] = []

    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"len_register={len(self.register_lst)}" + sep
        return to_return

    @abc.abstractmethod
    def register(
        self,
        lst_mu: Seq[DistributionT],
        pos_wgt: PosWgtT,
    ) -> RegisterValueT:
        """
        Get the register value from the distribution and
        position/weights.
        """
        raise NotImplementedError

    def __getitem__(self, item) -> RegisterValueT:
        return self.register_lst[item]

    @override
    def init_algorithm(self) -> tuple[Seq[DistributionT], PosWgtT]:
        result = super().init_algorithm()
        self.register_lst.append(self.register(*result))
        return result

    @override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[Seq[DistributionT], PosWgtT]:
        result = super().step_algorithm(k, pos_wgt_k)
        self.register_lst.append(self.register(*result))
        return result


class LogPosWgtIterProxy[DistributionT, PosWgtT](
    BaseLogProxy[DistributionT, PosWgtT, PosWgtT]
):
    """
    Proxy class for the SGDW algorithm to log the position and
    weights of the distribution at each iteration of the algorithm.
    """

    @override
    def register(
        self,
        lst_mu: Seq[DistributionT],
        pos_wgt: PosWgtT,
    ) -> PosWgtT:
        return pos_wgt


class LogDistrIterProxy[DistributionT, PosWgtT](
    BaseLogProxy[DistributionT, PosWgtT, DistributionT]
):
    """
    Proxy class for the SGDW algorithm to log distribution at
    each iteration of the algorithm.
    """

    @override
    def register(
        self,
        lst_mu: Seq[DistributionT],
        pos_wgt: PosWgtT,
    ) -> DistributionT:
        return self.create_distribution(pos_wgt)


class LogPosWgtSampledProxy[DistributionT, PosWgtT](
    BaseLogProxy[DistributionT, PosWgtT, Seq[PosWgtT]]
):
    """
    Proxy class for the SGDW algorithm to log the positions and
    weights of the sampled distributions at each iteration of the
    algorithm.
    """

    @override
    def register(
        self,
        lst_mu: Seq[DistributionT],
        pos_wgt: PosWgtT,
    ) -> Seq[PosWgtT]:
        return [self.get_pos_wgt(mu) for mu in lst_mu]


class LogDistrSampledProxy[DistributionT, PosWgtT](
    BaseLogProxy[DistributionT, PosWgtT, Seq[DistributionT]]
):
    """
    Proxy class for the SGDW algorithm to log the sampled
    distributions at each iteration of the algorithm.
    """

    @override
    def register(
        self,
        lst_mu: Seq[DistributionT],
        pos_wgt: PosWgtT,
    ) -> Seq[DistributionT]:
        return lst_mu


class SGDWProjectedDecorator[DistributionT, PosWgtT](
    SGDWBaseWrapper[DistributionT, PosWgtT]
):
    """
    Decorator class for the SGDW algorithm to project the position and
    weights of the distribution using a projector function, at each
    iteration of the algorithm.
    """

    def __init__(
        self,
        wrapee: SGDW[DistributionT, PosWgtT],
        projector: ProjectorFn[PosWgtT],
        project_every: int | None = 1,
    ) -> None:
        super().__init__(wrapee)
        self.projector = projector
        self.project_every = project_every

    def is_proj_iter(self) -> bool:
        """
        Check if the current iteration should be projected.
        """
        if self.project_every is None:
            return False
        k = self.iter_params.k
        proj_every = self.project_every
        return k % proj_every == proj_every - 1

    @override
    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"project_every={self.project_every}" + sep
        return to_return

    @override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[Seq[DistributionT], PosWgtT]:
        lst_mu_kp1, pos_wgt_kp1 = super().step_algorithm(k, pos_wgt_k)

        if self.is_proj_iter():
            pos_wgt_kp1 = self.projector(pos_wgt_kp1)

        return lst_mu_kp1, pos_wgt_kp1

    @override
    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        if self.project_every is not None:
            pos_wgt = self.projector(pos_wgt)
        return super().create_barycenter(pos_wgt)
