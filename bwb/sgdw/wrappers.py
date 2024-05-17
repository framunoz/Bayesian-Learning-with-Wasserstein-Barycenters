import abc
import typing as t
from copy import deepcopy
from datetime import timedelta

import bwb.logging_ as logging
from bwb.sgdw.sgdw import SGDW

__all__ = [
    "ReportProxy",
    "PosWgtIterRegProxy",
    "DistrIterRegisterProxy",
    "PosWgtSampledRegProxy",
    "DistrSampledRegProxy",
    "SGDWProjectedDecorator",
]

log = logging.get_logger(__name__)

type ProjectorFn[PosWgtT] = t.Callable[[PosWgtT], PosWgtT]


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

    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        wrapee_repr = self.wrapee._repr_(sep, tab, n_tab, new_line)
        to_return += wrapee_repr + sep
        return to_return

    @t.override
    @property
    def callback(self) -> t.Callable[[], None]:
        return self.wrapee.callback

    @callback.setter
    def callback(self, value: t.Callable[[], None]) -> None:
        self.wrapee.callback = value

    @t.override
    def first_sample(self) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        return self.wrapee.first_sample()

    @t.override
    def init_algorithm(self) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        return self.wrapee.init_algorithm()

    @t.override
    def update_pos_wgt(
        self,
        pos_wgt_k: PosWgtT,
        lst_mu_k: t.Sequence[DistributionT],
        gamma_k: float,
    ) -> PosWgtT:
        return self.wrapee.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

    @t.override
    def _compute_wass_dist(
        self, pos_wgt_k: PosWgtT, pos_wgt_kp1: PosWgtT, gamma_k: float
    ) -> float:
        return self.wrapee._compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)

    @t.override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        return self.wrapee.step_algorithm(k, pos_wgt_k)

    @t.override
    def get_pos_wgt(self, mu: DistributionT) -> PosWgtT:
        return self.wrapee.get_pos_wgt(mu)

    @t.override
    def create_distribution(self, pos_wgt: PosWgtT) -> DistributionT:
        return self.wrapee.create_distribution(pos_wgt)

    @t.override
    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        return self.wrapee.create_barycenter(pos_wgt)


class ReportOptions(t.TypedDict, total=False):
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

    def __init__(
        self,
        wrapee: SGDW[DistributionT, PosWgtT], *,
        report_every: int = 10,
        len_bar: int = 5,
        include_dict: ReportOptions = None,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(wrapee)
        self.report_every = report_every
        self.len_bar = len_bar
        self.include_dict = deepcopy(include_dict or self.INCLUDE_OPTIONS)
        self.level = level

    @t.override
    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"report_every={self.report_every}" + sep
        to_return += space + f"level={self.level}" + sep
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

    @t.override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        result = super().step_algorithm(k, pos_wgt_k)

        if self.is_report_iter():
            log.log(self.level, self.make_report())

        return result


class BaseRegisterProxy[DistributionT, PosWgtT, RegisterValueT](
    SGDWBaseWrapper[DistributionT, PosWgtT]
):
    register_lst: t.List[RegisterValueT]

    def __init__(self, wrapee: SGDW[DistributionT, PosWgtT]) -> None:
        super().__init__(wrapee)
        self.register_lst = []

    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"len_register={len(self.register_lst)}" + sep
        return to_return

    @abc.abstractmethod
    def get_register(
        self,
        lst_mu: t.Sequence[DistributionT],
        pos_wgt: PosWgtT,
    ) -> RegisterValueT:
        """
        Get the register value from the distribution and
        position/weights.
        """
        raise NotImplementedError

    def __getitem__(self, item) -> RegisterValueT:
        return self.register_lst[item]

    @t.override
    def init_algorithm(self) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        result = super().init_algorithm()
        self.register_lst.append(self.get_register(*result))
        return result

    @t.override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        result = super().step_algorithm(k, pos_wgt_k)
        self.register_lst.append(self.get_register(*result))
        return result


class PosWgtIterRegProxy[DistributionT, PosWgtT](
    BaseRegisterProxy[DistributionT, PosWgtT, PosWgtT]
):
    """
    Proxy class for the SGDW algorithm to register the position and
    weights of the distribution at each iteration of the algorithm.
    """

    @t.override
    def get_register(
        self,
        lst_mu: t.Sequence[DistributionT],
        pos_wgt: PosWgtT,
    ) -> PosWgtT:
        return pos_wgt


class DistrIterRegisterProxy[DistributionT, PosWgtT](
    BaseRegisterProxy[DistributionT, PosWgtT, DistributionT]
):
    """
    Proxy class for the SGDW algorithm to register the distribution at
    each iteration of the algorithm.
    """

    @t.override
    def get_register(
        self,
        lst_mu: t.Sequence[DistributionT],
        pos_wgt: PosWgtT,
    ) -> DistributionT:
        return self.create_distribution(pos_wgt)


class PosWgtSampledRegProxy[DistributionT, PosWgtT](
    BaseRegisterProxy[DistributionT, PosWgtT, t.Sequence[PosWgtT]]
):
    """
    Proxy class for the SGDW algorithm to register the positions and
    weights of the sampled distributions at each iteration of the
    algorithm.
    """

    @t.override
    def get_register(
        self,
        lst_mu: t.Sequence[DistributionT],
        pos_wgt: PosWgtT,
    ) -> t.Sequence[PosWgtT]:
        return [self.get_pos_wgt(mu) for mu in lst_mu]


class DistrSampledRegProxy[DistributionT, PosWgtT](
    BaseRegisterProxy[DistributionT, PosWgtT, t.Sequence[DistributionT]]
):
    """
    Proxy class for the SGDW algorithm to register the sampled
    distributions at each iteration of the algorithm.
    """

    @t.override
    def get_register(
        self,
        lst_mu: t.Sequence[DistributionT],
        pos_wgt: PosWgtT,
    ) -> t.Sequence[DistributionT]:
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
        project_every: int = 1,
    ) -> None:
        super().__init__(wrapee)
        self.projector = projector
        self.project_every = project_every

    def is_proj_iter(self) -> bool:
        """
        Check if the current iteration should be projected.
        """
        k = self.iter_params.k
        proj_every = self.project_every
        return k % proj_every == proj_every - 1

    @t.override
    def _additional_repr_(
        self, sep: str, tab: str, n_tab: int, new_line: str
    ) -> str:
        space = tab * n_tab
        to_return = super()._additional_repr_(sep, tab, n_tab, new_line)
        to_return += space + f"project_every={self.project_every}" + sep
        return to_return

    @t.override
    def step_algorithm(
        self,
        k: int,
        pos_wgt_k: PosWgtT
    ) -> tuple[t.Sequence[DistributionT], PosWgtT]:
        lst_mu_kp1, pos_wgt_kp1 = super().step_algorithm(k, pos_wgt_k)

        if self.is_proj_iter():
            pos_wgt_kp1 = self.projector(pos_wgt_kp1)

        return lst_mu_kp1, pos_wgt_kp1

    @t.override
    def create_barycenter(self, pos_wgt: PosWgtT) -> DistributionT:
        pos_wgt = self.projector(pos_wgt)
        return super().create_barycenter(pos_wgt)
