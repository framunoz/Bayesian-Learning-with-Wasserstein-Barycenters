import abc
import numbers as num
import time
import typing as t
import warnings
from datetime import timedelta

import ot
import torch
from torch import linalg as LA

import bwb.bregman
import bwb.distributions as dist
import bwb.transports as tpt
from bwb import logging, utils
from bwb.utils import _DistributionT
from wgan_gp.wgan_gp_vae.utils import ProjectorOnManifold

__all__ = [
    "Gamma",
    "BaseSGDW",
    "DiscreteDistributionSGDW",
    "DistributionDrawSGDW",
    "ConvDistributionDrawSGDW",
    "DebiesedDistributionDrawSGDW",
]

_log = logging.get_logger(__name__)
_bar = "=" * 5


def gamma(*, a: float = 1, b: float = 0, c: float = 1):
    """
    This function returns a gamma function with parameters a, b, and c.

    Parameters:
        a (float): The shape parameter of the gamma distribution. Default is 1.
        b (float): The scale parameter of the gamma distribution. Default is 0.
        c (float): The location parameter of the gamma distribution. Default is 1.

    Returns:
        function: A gamma function that takes a single parameter k and returns the value of the gamma function at k.
    """

    def _gamma(k):
        return a / (b ** (1 / c) + k) ** c

    return _gamma


class Gamma:
    """
    This class represents a gamma function.

    Parameters:
    - a (float): The parameter 'a' of the gamma function.
    - b (float): The parameter 'b' of the gamma function.
    - c (float): The parameter 'c' of the gamma function.
    """

    def __init__(self, a=1, b=0, c=1):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, k):
        """
        Compute the value of the gamma function for a given input.

        Parameters:
        - k (float): The input value.

        Returns:
        - float: The value of the gamma function for the given input.
        """
        return self.a / (self.b ** (1 / self.c) + k) ** self.c


class Schedule:
    """
    This class contains the schedule for the learning rate and batch size.
    """

    def __init__(
        self,
        step_schedule: t.Callable[[int], float],
        batch_size: t.Callable[[int], int] | int,
    ):
        self.step_schedule = step_schedule
        self.batch_size = batch_size

    @property
    def step_schedule(self) -> t.Callable[[int], float]:
        """The step schedule for the algorithm."""
        return self._learning_rate

    @step_schedule.setter
    def step_schedule(self, learning_rate):
        # Check if learning_rate is callable
        if not callable(learning_rate):
            raise TypeError("learning_rate must be a callable")

        # Check if learning_rate is callable that accepts an integer and returns a float
        try:
            if not isinstance(learning_rate(1), float):
                raise ValueError("learning_rate must return a float")
        except Exception as e:
            raise ValueError("learning_rate must accept an integer argument") from e

        self._learning_rate: t.Callable[[int], float] = learning_rate

    @property
    def batch_size(self) -> t.Callable[[int], int]:
        """The batch size schedule for the algorithm."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        # Check if batch_size is callable or an integer
        if not callable(batch_size) and not isinstance(batch_size, int):
            raise TypeError("batch_size must be a callable or an int")

        # If batch_size is an integer, convert it to a callable
        if isinstance(batch_size, int):
            aux: int = batch_size

            def batch_size(n: int):
                return aux

        # Check if batch_size is callable that accepts an integer and returns an integer
        try:
            if not isinstance(batch_size(0), int):
                raise ValueError("batch_size must return an integer")
        except Exception as e:
            raise ValueError("batch_size must accept an integer argument") from e

        self._batch_size: t.Callable[[int], int] = batch_size


class DetentionParameters:
    """
    This class contains the detention parameters for the algorithm.
    """

    def __init__(
        self, tol: float = 1e-8, max_iter: int = 100_000, max_time: float = float("inf")
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.max_time = max_time

    @property
    def tol(self) -> float:
        """
        The tolerance value for convergence.
        """
        return self._tol

    @tol.setter
    def tol(self, tol: float):
        if not isinstance(tol, num.Real):
            raise TypeError("tol must be a real number")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        self._tol: float = float(tol)

    @property
    def max_iter(self) -> int:
        """
        The maximum number of iterations.
        """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, max_iter: int):
        if not (isinstance(max_iter, num.Integral) or max_iter == float("inf")):
            raise TypeError("max_iter must be an integer or infinity")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self._max_iter: int = int(max_iter)

    @property
    def max_time(self) -> float:
        """
        The maximum time allowed for the algorithm to run.
        """
        return self._max_time

    @max_time.setter
    def max_time(self, max_time: float):
        if not isinstance(max_time, num.Real):
            raise TypeError("max_time must be a real number")
        if max_time <= 0:
            raise ValueError("max_time must be positive")
        self._max_time: float = float(max_time)

    def __repr__(self) -> str:
        time_fmt = "∞"
        if self.max_time != float("inf"):
            max_time = round(self.max_time)
            time_fmt = str(timedelta(seconds=max_time))
        max_iter_fmt = f"{self.max_iter:_}" if self.max_iter != float("inf") else "∞"

        return f"DetentionParameters(tol={self.tol:.2e}, max_iter={max_iter_fmt}, max_time={time_fmt})"


class IterationParameters:
    """
    This class contains the iteration parameters for the algorithm.
    """

    def __init__(self, det_params: DetentionParameters):
        self.det_params = det_params
        """The detention parameters for the algorithm."""
        self.k = 0
        """The iteration number."""
        self.tic = time.time()
        """The start time of the algorithm."""
        self.tic_ = time.time()
        """The start time of the iteration."""
        self.toc = time.time()
        """The end time of the iteration."""
        self.diff_t = 0
        """The time difference between tic and toc."""
        self.w_dist = float("inf")
        """The Wasserstein distance."""

    def init_params(self):
        """
        Initializes the iteration metrics.

        This method sets the initial values for the iteration metrics used in the class.
        It initializes the following attributes:
        - k: The iteration number (initialized to 0).
        - tic: The start time of the algorithm (initialized to the current time).
        - tic_: The start time of the iteration (initialized to the current time).
        - toc: The end time of the iteration (initialized to the current time).
        - diff_t: The time difference between tic_ and toc.
        - w_dist: The Wasserstein distance (initialized to infinity).
        """
        self.k = 0
        self.tic = time.time()
        self.tic_ = time.time()
        self.toc = time.time()
        self.diff_t = 0
        self.w_dist = float("inf")

    def start_iteration(self):
        """
        Start the iteration.

        This method starts the iteration and sets the start time of the iteration to the current time.
        """
        self.tic_ = time.time()

    def update_iteration(self):
        """
        Update the iteration metrics.

        This method updates the iteration metrics used in the class.
        It updates the following attributes:
        - k: The iteration number (incremented by 1).
        - tic_: The start time of the iteration (set to the current time).
        - toc: The end time of the iteration (set to the current time).
        - diff_t: The time difference between tic_ and toc.

        :param tic_: The start time of the iteration.
        :type tic_: float
        :return: None
        """
        self.k += 1
        self.toc = time.time()
        self.diff_t = self.toc - self.tic_

    def detention_criteria(self) -> bool:
        """
        Determines the detention criteria for the algorithm.

        :return: True if the detention criteria is met, False otherwise.
        """
        return (
            self.k >= self.det_params.max_iter  # Reaches maximum iteration
            or self.toc - self.tic >= self.det_params.max_time  # Reaches maximum time
            or self.w_dist < self.det_params.tol  # Achieves convergence in distance
        )

    def __repr__(self) -> str:
        w_dist_fmt = f"{self.w_dist:.6f}" if self.w_dist != float("inf") else "∞"
        return f"IterationParameters(k={self.k}, w_dist={w_dist_fmt}, t={self.toc - self.tic:.2f} [seg], Δt={self.diff_t * 1000:.2f} [ms])"


class History(t.Generic[_DistributionT]):
    """
    This class contains the history logic of the algorithm.
    """

    def __init__(
        self,
        pos_wgt: bool = False,
        distr: bool = False,
        distr_samp: bool = False,
    ):
        self.pos_wgt = pos_wgt
        self.distr = distr
        self.distr_samp = distr_samp

    @property
    def create_distr(self) -> t.Callable[[t.Any], _DistributionT]:
        """The function to create a distribution from the position and weight."""
        return self._create_distribution

    @create_distr.setter
    def create_distr(self, create_distribution):
        if not callable(create_distribution):
            raise TypeError("create_distribution must be a callable")
        self._create_distribution: t.Callable[[t.Any], _DistributionT] = (
            create_distribution
        )

    @property
    def pos_wgt(self) -> list[t.Any]:
        """Whether to store the history of the position and weights at each iteration."""
        if isinstance(self._pos_wgt, bool):
            return []
        return self._pos_wgt

    @pos_wgt.setter
    def pos_wgt(self, pos_wgt_hist):
        if not (isinstance(pos_wgt_hist, bool) or isinstance(pos_wgt_hist, list)):
            raise TypeError("pos_wgt_hist must be a boolean or a list")
        self._pos_wgt = pos_wgt_hist

    @property
    def distr(self) -> list[_DistributionT]:
        """Whether to store the history of the distributions at each iteration."""
        if isinstance(self._distr, bool):
            return []
        return self._distr

    @distr.setter
    def distr(self, distr_hist):
        if not (isinstance(distr_hist, bool) or isinstance(distr_hist, list)):
            raise TypeError("distr_hist must be a boolean or a list")
        self._distr = distr_hist

    @property
    def distr_samp(self) -> list[list[_DistributionT]]:
        """Whether to store the history of the distributions sampled by the sampler at each iteration."""
        if isinstance(self._distr_samp, bool):
            return []
        return self._distr_samp

    @distr_samp.setter
    def distr_samp(self, distr_samp_hist):
        if not (isinstance(distr_samp_hist, bool) or isinstance(distr_samp_hist, list)):
            raise TypeError("distr_samp_hist must be a boolean or a list")
        self._distr_samp = distr_samp_hist

    def set_params(
        self,
        pos_wgt: bool = None,
        distr: bool = None,
        distr_samp: bool = None,
        create_distribution: t.Callable[[t.Any], _DistributionT] = None,
    ):
        self.pos_wgt = pos_wgt if pos_wgt is not None else self.pos_wgt
        self.distr = distr if distr is not None else self.distr
        self.distr_samp = distr_samp if distr_samp is not None else self.distr_samp
        self.create_distr = (
            create_distribution
            if create_distribution is not None
            else self.create_distr
        )

    def init_histories(self, mu_0, pos_wgt_0):
        """
        Initialize the histories for the algorithm.

        This method should initialize the histories for the algorithm.

        :param mu_0: The first sample from the distribution sampler.
        :param pos_wgt_0: The position and weight that come from the first sample.
        """
        if self._pos_wgt:
            self.pos_wgt = [pos_wgt_0]
        if self._distr:
            self.distr = [self.create_distr(pos_wgt_0)]
        if self._distr_samp:
            self.distr_samp = [[mu_0]]

    def update_pos_wgt(self, pos_wgt_kp1):
        """
        Update the position and weight for the next iteration.

        This method should update the position and weight for the next iteration.

        :param pos_wgt_kp1: The position and weight that come from the next sample.
        """
        if self.pos_wgt:
            self.pos_wgt.append(pos_wgt_kp1)

    def update_distr(self, pos_wgt_kp1):
        """
        Update the distribution history for the next iteration.

        This method should update the distribution history for the next iteration.

        :param distr_kp1: The distribution created from the next sample.
        """
        if self.distr:
            self.distr.append(self.create_distr(pos_wgt_kp1))

    def update_distr_samp(self, lst_mu_k):
        """
        Update the distribution sampler history for the next iteration.

        This method should update the distribution sampler history for the next iteration.

        :param lst_mu_k: The list of distributions sampled by the sampler at the current iteration.
        """
        if self.distr_samp:
            self.distr_samp.append(lst_mu_k)

    def update_histories(self, pos_wgt_kp1, lst_mu_k):
        """
        Update the histories for the next iteration.

        This method should update the histories for the next iteration.

        :param pos_wgt_kp1: The position and weight that come from the next sample.
        :param lst_mu_k: The list of distributions sampled by the sampler at the current iteration.
        """
        self.update_pos_wgt(pos_wgt_kp1)
        self.update_distr(pos_wgt_kp1)
        self.update_distr_samp(lst_mu_k)

    def has_pos_wgt(self) -> bool:
        if isinstance(self._pos_wgt, bool):
            return self._pos_wgt
        if self.pos_wgt:
            return True
        return False

    def has_distr(self) -> bool:
        if isinstance(self._distr, bool):
            return self._distr
        if self.distr:
            return True
        return False

    def has_distr_samp(self) -> bool:
        if isinstance(self._distr_samp, bool):
            return self._distr_samp
        if self.distr_samp:
            return True
        return False

    def has_histories(self):
        return self.has_pos_wgt() or self.has_distr() or self.has_distr_samp()

    def __getitem__(self, key: t.Literal["pos_wgt", "distr", "distr_samp"]):
        match key:
            case "pos_wgt":
                return self.pos_wgt
            case "distr":
                return self.distr
            case "distr_samp":
                return self.distr_samp
            case _:
                raise KeyError(f"Invalid key: {key}")

    def __repr__(self) -> str:
        return f"History(pos_wgt={self.has_pos_wgt()}, distr={self.has_distr()}, distr_samp={self.has_distr_samp()}, len={len(self)})"

    def __len__(self) -> int:
        if self.has_pos_wgt():
            return len(self.pos_wgt)
        if self.has_distr():
            return len(self.distr)
        if self.has_distr_samp():
            return len(self.distr_samp)
        return 0


class Report:
    """
    This class contains the report logic of the algorithm.
    """

    def __init__(
        self,
        iter_params: IterationParameters,
        report_every: int = 10,
        include_iter: bool = True,
        include_w_dist: bool = False,
        include_lr: bool = True,
        include_time: bool = False,
        len_bar: int = 5,
    ):
        self.iter_params = iter_params
        self.report_every = report_every
        self.include_iter = include_iter
        self.include_w_dist = include_w_dist
        self.include_lr = include_lr
        self.include_time = include_time
        self.len_bar = len_bar

    def set_params(
        self,
        include_iter: bool = None,
        include_w_dist: bool = None,
        include_lr: bool = None,
        include_time: bool = None,
        len_bar: int = None,
    ):
        self.include_iter = (
            include_iter if include_iter is not None else self.include_iter
        )
        self.include_w_dist = (
            include_w_dist if include_w_dist is not None else self.include_w_dist
        )
        self.include_lr = include_lr if include_lr is not None else self.include_lr
        self.include_time = (
            include_time if include_time is not None else self.include_time
        )
        self.len_bar = len_bar if len_bar is not None else self.len_bar

    def make_report(
        self,
        gamma_k: float,
    ):
        """
        Generate a report for the algorithm.
        """
        bar = "=" * self.len_bar

        report = bar + " "

        if self.include_iter:
            report += f"k = {self.iter_params.k}, "

        if self.include_w_dist:
            report += f"Wass. dist. = {self.iter_params.w_dist:.6f}, "

        if self.include_lr:
            report += f"gamma_k = {gamma_k:.4%}, "

        if self.include_time:
            total_time = round(self.iter_params.toc - self.iter_params.tic)
            time_fmt = str(timedelta(seconds=total_time))
            report += f"t = {time_fmt}, "
            report += f"Δt = {(self.iter_params.diff_t) * 1000:.2f} [ms], "
            report += f"Δt per iter. = {(self.iter_params.toc - self.iter_params.tic) * 1000 / (self.iter_params.k + 1):.2f} [ms/iter], "

        report = report[:-2] + " " + bar

        return report

    def is_report_iter(self) -> bool:
        return self.iter_params.k % self.report_every == 0


class BaseSGDW(t.Generic[_DistributionT], metaclass=abc.ABCMeta):
    """
    Base class for Stochastic Gradient Descent in Wasserstein Space.

    This class provides a base implementation for Stochastic Gradient Descent in Wasserstein Space.
    It defines the common attributes and methods used by the derived classes.

    :param learning_rate: A callable function that takes an integer argument (k) and returns the learning rate (\gamma_k) for iteration k.
    :type learning_rate: callable
    :param batch_size: A callable function that takes an integer argument (k) and returns the batch size (S_k) for iteration k. Alternatively, it can be a constant integer value.
    :type batch_size: callable or int
    :param tol: The tolerance value for convergence. Defaults to 1e-8.
    :type tol: float
    :param max_iter: The maximum number of iterations. Defaults to 100_000.
    :type max_iter: int
    :param max_time: The maximum time allowed for the algorithm to run. Defaults to infinity.
    :type max_time: float
    :param report_every: The frequency at which to report the metrics. Defaults to 10.
    :type report_every: int

    :raises TypeError: If learning_rate is not a callable or batch_size is not a callable or an integer.
    :raises ValueError: If learning_rate does not return a float or batch_size does not return an integer.
    """

    def __init__(
        self,
        distr_sampler: dist.DistributionSampler[_DistributionT],
        learning_rate,  # The \gamma_k schedule
        batch_size,  # The S_k schedule
        tol: float = 1e-8,  # Tolerance to converge
        max_iter: int = 100_000,  # Maximum number of iterations
        max_time: float = float("inf"),  # Maximum time in seconds
        report_every: int = 10,  # Report every k iterations
    ):
        # Distribution sampler
        self.distr_sampler = distr_sampler
        """The distribution sampler for the algorithm."""

        # Schedule parameters
        self.schd = Schedule(learning_rate, batch_size)

        # Detention parameters
        self.det_params = DetentionParameters(tol, max_iter, max_time)

        # Iteration metrics
        self.iter_params = IterationParameters(self.det_params)

        # History of the position and weights
        self.hist: History[_DistributionT] = History()

        # Report parameters
        self.report = Report(self.iter_params, report_every=report_every)

        # Values for dtype and device
        mu = self.distr_sampler.draw()
        self.dtype = mu.dtype
        """The data type for the algorithm. Defaults to the data type of the first distribution drawn from the sampler."""
        self.device = mu.device
        """The device for the algorithm. Defaults to the device of the first distribution drawn from the sampler."""
        self.val = torch.tensor(1, dtype=self.dtype, device=self.device)
        """A tensor with value 1, used to pass to the device."""

    def set_schedules(
        self,
        step_schedule: t.Callable[[int], float],  # The \gamma_k schedule
        batch_size: t.Union[t.Callable[[int], int], int],  # The S_k schedule
    ):
        """
        Set the step and batch size schedules for the algorithm.

        :param step_schedule: A callable function that takes an integer argument (k) and returns the step (\gamma_k) for iteration k.
        :param batch_size: A callable function that takes an integer argument (k) and returns the batch size (S_k) for iteration k. Alternatively, it can be a constant integer value.
        :raises TypeError: If learning_rate is not a callable or batch_size is not a callable or an integer.
        :raises ValueError: If learning_rate does not return a float or batch_size does not return an integer.
        :return: The object itself.
        """
        self.schd.step_schedule = step_schedule
        self.schd.batch_size = batch_size
        return self

    def set_detention_params(
        self, tol: float = 1e-8, max_iter: int = 100_000, max_time: float = float("inf")
    ):
        """
        Set the detention parameters for the model.

        :param tol: The tolerance value for convergence. Defaults to 1e-8.
        :type tol: float
        :param max_iter: The maximum number of iterations. Defaults to 100_000.
        :type max_iter: int
        :param max_time: The maximum time allowed for the algorithm to run. Defaults to infinity.
        :type max_time: float
        :return: The object itself.
        :raises TypeError: If tol is not a real number, max_iter is not an integer or max_time is not a real number.
        :raises ValueError: If tol is not positive, max_iter is not positive or max_time is not positive.
        """
        self.det_params.tol = tol
        self.det_params.max_iter = max_iter
        self.det_params.max_time = max_time
        return self

    @abc.abstractmethod
    def _create_distribution(self, pos_wgt) -> _DistributionT:
        """
        Create a distribution from the position and weight.

        This method should create a distribution from the position and weight and return it.

        :param pos_wgt: The position and weight.
        :return: The distribution created from the position and weight.
        """
        pass

    @abc.abstractmethod
    def first_sample(self) -> tuple[_DistributionT, t.Any]:
        """
        Draw the first sample from the distribution sampler. This corresponds to the first step of the algorithm.

        This method should draw the first sample from the distribution sampler and return it.

        :return: The first sample from the distribution sampler and the position and weight that come from the sample.
        """
        pass

    def samp_distributions(self, S_k: int) -> t.Sequence[_DistributionT]:
        """
        Draw S_k samples from the distribution sampler.

        This method should draw S_k samples from the distribution sampler and return them.

        :param S_k: The batch size for the current iteration.
        :return: A list of distributions drawn from the sampler.
        """
        return self.distr_sampler.rvs(S_k)

    @abc.abstractmethod
    def update_pos_wgt(self, pos_wgt_k, lst_mu_k, gamma_k) -> t.Any:
        """
        Update the position and weight for the next iteration.

        This method should update the position and weight for the next iteration and return it.

        :param pos_wgt_k: The position and weight that come from the current sample.
        :param lst_mu_k: The list of distributions drawn from the sampler at the current iteration.
        :param gamma_k: The learning rate for the next sample.
        """
        pass

    def compute_wass_dist(self, pos_wgt_k, pos_wgt_kp1, gamma_k):
        """
        Compute the Wasserstein distance between two positions and weights.

        This method should compute the Wasserstein distance between two positions and weights and return it.

        :param pos_wgt_k: The position and weight that come from the current sample.
        :param pos_wgt_kp1: The position and weight that come from the next sample.
        :param gamma_k: The learning rate for the next sample.
        """
        pass

    def run(
        self,
        pos_wgt_hist=False,
        distr_hist=False,
        distr_samp_hist=False,
        include_iter=None,
        include_w_dist=None,
        include_lr=None,
        include_time=None,
    ):
        """
        Run the algorithm.

        This method runs the algorithm and returns the final position weights and optional history data.

        :param include_iter: Whether to include the iteration number in the report.
        :param include_w_dist: Whether to include the Wasserstein distance in the report.
        :param include_lr: Whether to include the learning rate in the report.
        :param include_time: Whether to include the iteration time in the report.
        :return: A tuple containing the final position weights and optional history data.
        """
        self.hist.set_params(
            pos_wgt=pos_wgt_hist,
            distr=distr_hist,
            distr_samp=distr_samp_hist,
            create_distribution=self._create_distribution,
        )

        self.report.set_params(
            include_iter=include_iter,
            include_w_dist=include_w_dist,
            include_lr=include_lr,
            include_time=include_time,
        )

        # Step 1: Sampling a mu_0. For convention, we use mu_k to denote the current sample
        mu_k, pos_wgt_k = self.first_sample()

        self.hist.init_histories(mu_k, pos_wgt_k)

        self.iter_params.init_params()

        while not self.iter_params.detention_criteria():
            # Time at the beginning of the iteration
            self.iter_params.start_iteration()

            k = self.iter_params.k

            # Step 2: Draw S_k samples from the distribution sampler
            S_k = self.schd.batch_size(k)
            lst_mu_k = self.samp_distributions(S_k)

            # Step 3: Compute the distribution of mu_{k+1}
            gamma_k = self.schd.step_schedule(k)
            pos_wgt_kp1 = self.update_pos_wgt(pos_wgt_k, lst_mu_k, gamma_k)

            # Step 4 (optional): Compute the Wasserstein distance
            self.compute_wass_dist(pos_wgt_k, pos_wgt_kp1, gamma_k)

            # Step 5: Add to history
            self.hist.update_histories(pos_wgt_kp1, lst_mu_k)

            # If the iteration is a multiple of report_every, print the report
            if self.report.is_report_iter():
                _log.info(self.report.make_report(gamma_k))

            # Step 6: Update
            pos_wgt_k = pos_wgt_kp1
            self.iter_params.update_iteration()

        barycenter = self._create_distribution(pos_wgt_k)

        if self.hist.has_histories():
            return barycenter, self.hist

        return barycenter


class DiscreteDistributionSGDW(BaseSGDW[dist.DiscreteDistribution]):
    def __init__(
        self,
        transport: tpt.BaseTransport,
        distr_sampler: dist.DistributionSampler[dist.DiscreteDistribution],
        learning_rate: t.Callable[[int], float],
        batch_size: t.Union[t.Callable[[int], int], int],
        alpha: float = 1.0,
        tol: float = 1e-8,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
        report_every=10,
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            learning_rate=learning_rate,
            batch_size=batch_size,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
            report_every=report_every,
        )
        self.transport = transport
        self.alpha = alpha
        self.include_w_dist = True

    def _create_distribution(self, pos_wgt) -> dist.DiscreteDistribution:
        X_k, m = pos_wgt
        return dist.DiscreteDistribution(support=X_k, weights=m)

    def first_sample(
        self,
    ) -> tuple[dist.DiscreteDistribution, t.Tuple[torch.Tensor, torch.Tensor]]:
        mu_0: dist.DiscreteDistribution = self.distr_sampler.draw()
        X_k, m = utils.partition(
            X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=self.alpha
        )
        X_k, m = X_k.to(self.val), m.to(self.val)
        return mu_0, (X_k, m)

    def update_pos_wgt(
        self, pos_wgt_k, lst_mu_k, gamma_k
    ) -> t.Tuple[torch.Tensor, torch.Tensor]:
        X_k, m = pos_wgt_k
        T_X_k = torch.zeros_like(X_k, dtype=self.dtype, device=self.device)
        S_k = len(lst_mu_k)
        for mu_i_k in lst_mu_k:
            X_i_k, m_i_k = mu_i_k.enumerate_nz_support_(), mu_i_k.nz_probs
            X_i_k, m_i_k = X_i_k.to(self.val), m_i_k.to(self.val)
            m_i_k /= torch.sum(m_i_k)
            self.transport.fit(Xs=X_k, mu_s=m, Xt=X_i_k, mu_t=m_i_k)
            T_X_k += self.transport.transform(X_k)
        T_X_k /= S_k
        X_kp1 = (1 - gamma_k) * X_k + gamma_k * T_X_k
        return (X_kp1, m)

    def compute_wass_dist(self, pos_wgt_k, pos_wgt_kp1, gamma_k):
        X_k, m = pos_wgt_k
        X_kp1, m = pos_wgt_kp1
        diff = X_k - X_kp1
        w_dist = float((gamma_k**2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))
        self.w_dist = w_dist


class DistributionDrawSGDW(BaseSGDW[dist.DistributionDraw], metaclass=abc.ABCMeta):
    def __init__(
        self,
        distr_sampler: dist.DistributionSampler[dist.DistributionDraw],
        learning_rate: t.Callable[[int], float],
        projector: t.Optional[ProjectorOnManifold] = None,
        tol: float = 0,
        max_iter: int = 100_000,
        max_time: float = float("inf"),
        report_every=10,
    ):
        super().__init__(
            distr_sampler=distr_sampler,
            learning_rate=learning_rate,
            batch_size=1,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
            report_every=report_every,
        )
        self.projector = projector
        self.set_geodesic_params()

    def _create_distribution(self, pos_wgt) -> dist.DistributionDraw:
        gs_weights_k = pos_wgt
        return dist.DistributionDraw.from_grayscale_weights(gs_weights_k)

    def first_sample(
        self,
    ) -> tuple[dist.DistributionDraw, torch.Tensor]:
        mu_0: dist.DistributionDraw = self.distr_sampler.draw()
        return mu_0, mu_0.grayscale_weights

    @abc.abstractmethod
    def _compute_geodesic(self, gs_weights_k, gs_weights_mu_k, gamma_k) -> torch.Tensor:
        """
        Compute the geodesic between two points in the Wasserstein space.

        :param gs_weights_k: The weights of the first point.
        :type gs_weights_k: torch.Tensor
        :param gs_weights_mu_k: The weights of the second point.
        :type gs_weights_mu_k: torch.Tensor
        :param gamma_k: The parameter controlling the interpolation between the two points.
        :type gamma_k: float

        :return: The geodesic between the two points.
        """
        ...

    def set_geodesic_params(
        self,
        reg=3e-3,
        method="sinkhorn",
        numItermax=1_000,
        stopThr=1e-8,
        verbose=False,
        warn=False,
        **kwargs,
    ):
        """
        Set the parameters for geodesic computation.

        Parameters:
        - reg (float): Regularization term for Sinkhorn algorithm. Default is 3e-3.
        - method (str): Method to use for geodesic computation. Default is "sinkhorn".
        - numItermax (int): Maximum number of iterations for Sinkhorn algorithm. Default is 1000.
        - stopThr (float): Stopping threshold for Sinkhorn algorithm. Default is 1e-8.
        - verbose (bool): Whether to print verbose output during computation. Default is False.
        - warn (bool): Whether to display warning messages. Default is False.
        - **kwargs: Additional keyword arguments to be passed to the geodesic computation method.

        Returns:
        - self: The current instance of the class.
        """
        self.conv_bar_kwargs = dict(
            reg=reg,
            method=method,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=False,
            warn=warn,
            **kwargs,
        )
        return self

    def update_pos_wgt(self, pos_wgt_k, lst_mu_k, gamma_k) -> torch.Tensor:
        gs_weights_k = pos_wgt_k
        mu_k = lst_mu_k[0]
        gs_weights_kp1 = self._compute_geodesic(
            gs_weights_k, mu_k.grayscale_weights, gamma_k
        )
        if self.projector is not None:
            gs_weights_kp1 = self.projector(gs_weights_kp1).to(self.val)
        return gs_weights_kp1


class ConvDistributionDrawSGDW(DistributionDrawSGDW):
    def _compute_geodesic(self, gs_weights_k, gs_weights_mu_k, gamma_k) -> torch.Tensor:
        return bwb.bregman.convolutional_barycenter2d(
            A=torch.stack([gs_weights_k, gs_weights_mu_k]),
            weights=torch.as_tensor(
                [1 - gamma_k, gamma_k], dtype=self.dtype, device=self.device
            ),
            **self.conv_bar_kwargs,
        )


class DebiesedDistributionDrawSGDW(DistributionDrawSGDW):
    def _compute_geodesic(self, gs_weights_k, gs_weights_mu_k, gamma_k) -> torch.Tensor:
        return ot.bregman.convolutional_barycenter2d_debiased(
            A=torch.stack([gs_weights_k, gs_weights_mu_k]),
            weights=torch.as_tensor(
                [1 - gamma_k, gamma_k], dtype=self.dtype, device=self.device
            ),
            **self.conv_bar_kwargs,
        )


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
        "This function is deprecated. Use the DiscreteDistributionSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(batch_size, int):
        aux = batch_size

        def batch_size(n):
            return aux

    batch_size: t.Callable[[int], int]

    # Paso 1: Sampling a mu_0
    mu_0: dist.DiscreteDistribution = distrib_sampler.draw()
    dtype, device = mu_0.dtype, mu_0.device
    _log.info(f"{dtype = }, {device = }")

    # Compute locations through the partition
    _log.info("Computing initial weights")
    tic = time.time()
    X_k, m = utils.partition(
        X=mu_0.enumerate_nz_support_(), mu=mu_0.nz_probs, alpha=alpha
    )
    X_k, m = X_k.to(dtype=dtype, device=device), m.to(dtype=dtype, device=device)
    toc = time.time()
    _log.info(f"total init weights: {len(m)}, Δt = {toc - tic:.2f} [seg]")

    # Create histories
    if position_history:
        position_history = [X_k]
    if distribution_history:
        distribution_history = [dist.DiscreteDistribution(support=X_k, weights=m)]
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
                _bar + f" {k = }, "
                f"w_dist = {w_dist:.4f}, "
                f"t = {toc - tic:.2f} [seg], "
                f"Δt = {diff_t * 1000:.2f} [ms], "
                f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} [ms/iter] " + _bar
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
            t_X_i_k, t_m_i_k = t_mu_i_k.enumerate_nz_support_(), t_mu_i_k.nz_probs
            # Pass to device
            t_X_i_k, t_m_i_k = t_X_i_k.to(dtype=dtype, device=device), t_m_i_k.to(
                dtype=dtype, device=device
            )
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
        w_dist = float((gamma_k**2) * torch.sum(m * LA.norm(diff, dim=1) ** 2))

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
        "This function is deprecated. Use the DistributionDrawSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Paso 1: Sampling a mu_0
    mu_k: dist.DistributionDraw = distrib_sampler.draw()
    dtype, device = mu_k.dtype, mu_k.device
    _log.info(f"{dtype = }, {device = }")

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
                _bar + f" {k = }, "
                f"t = {toc - tic:.2f} [seg], "
                f"Δt = {diff_t * 1000:.2f} [ms], "
                f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} [ms/iter] " + _bar
            )

        m_k: dist.DistributionDraw = distrib_sampler.draw()
        if distrib_sampler_history:
            distrib_sampler_history.append([m_k])

        # Compute the distribution of mu_{k+1}
        gamma_k = learning_rate(k)

        gs_weights_kp1, _ = bwb.bregman.convolutional_barycenter2d(
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
            mu_kp1 = dist.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
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
        "This function is deprecated. Use the DistributionDrawSGDW class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Paso 1: Sampling a mu_0
    mu_k: dist.DistributionDraw = distrib_sampler.draw()
    dtype, device = mu_k.dtype, mu_k.device
    _log.info(f"{dtype = }, {device = }")

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
                _bar + f" {k = }, "
                f"t = {toc - tic:.2f} [seg], "
                f"Δt = {diff_t * 1000:.2f} [ms], "
                f"Δt per iter. = {(toc - tic) * 1000 / (k + 1):.2f} [ms/iter] " + _bar
            )

        m_k: dist.DistributionDraw = distrib_sampler.draw()
        if distrib_sampler_history:
            distrib_sampler_history.append([m_k])

        # Compute the distribution of mu_{k+1}
        gamma_k = learning_rate(k)
        _log.debug(f"{gamma_k = :.6f}")

        gs_weights_kp1, _ = bwb.bregman.convolutional_barycenter2d(
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
            mu_kp1 = dist.DistributionDraw.from_grayscale_weights(gs_weights_kp1)
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
