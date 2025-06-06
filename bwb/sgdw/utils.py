"""
This module contains utility functions and classes for the algorithm.
"""
import numbers as num
import time
from datetime import timedelta
from typing import Callable, Iterator

import bwb.logging_ as logging

__all__ = [
    "step_scheduler",
    "StepSchedulerFn",
    "StepSchedulerArg",
    "BatchSizeFn",
    "BatchSizeArg",
    "Schedule",
    "DetentionParameters",
    "IterationParameters",
]

_log = logging.get_logger(__name__)

type StepSchedulerFn = Callable[[int], float]
type StepSchedulerArg = float | StepSchedulerFn
type BatchSizeFn = Callable[[int], int]
type BatchSizeArg = int | BatchSizeFn


def step_scheduler(
    *,
    a: float = 1,
    b: float = 0,
    c: float = 1
) -> StepSchedulerFn:
    r"""
    This function returns a step scheduler with parameters :math:`a`,
    :math:`b`, and :math:`c`.

    The formula for the step scheduler is given by:
    .. math::
        \gamma_k = \frac{a}{(b^{1/c} + k)^c}

    :param a: The scale parameter of the gamma distribution. Default is 1.
    :param b: The location parameter of the gamma distribution. Default is 0.
    :param c: The shape parameter of the gamma distribution. Default is 1.
    :return: A step scheduler that takes a single parameter :math:`k`
        and returns the value of the step scheduler at :math:`k`.
    """

    def _step_scheduler(k: int) -> float:
        r"""
        Compute the value of the step scheduler for a given input.

        The formula for the step scheduler is given by:
        .. math::
            \gamma_k = \frac{a}{(b^{1/c} + k)^c}

        :param k: The input value.
        :return: The value of the step scheduler for the given input.
        """
        return a / (b ** (1 / c) + k) ** c

    return _step_scheduler


class Schedule:
    """
    This class contains the schedule for the learning rate and batch size.
    """

    def __init__(
        self,
        step_schedule: StepSchedulerArg,
        batch_size: BatchSizeArg,
    ):
        self.step_schedule = step_schedule
        self.batch_size = batch_size

    @property
    def step_schedule(self) -> StepSchedulerFn:
        """The step schedule for the algorithm."""
        return self._step_schedule

    @step_schedule.setter
    def step_schedule(self, step_schd: StepSchedulerArg) -> None:
        # Check if step_schedule is callable or a float
        if not callable(step_schd) and not isinstance(step_schd, float):
            raise TypeError("step_schedule must be a callable or a float")

        # If step_schedule is a float, convert it to a callable
        if isinstance(step_schd, float):
            # Check if step_schd is between 0 and 1
            if not 0 <= step_schd <= 1:
                raise ValueError("step_schedule must be between 0 and 1")

            aux: float = step_schd

            # noinspection PyUnusedLocal
            def step_schd(n: int) -> float:
                """Return the learning rate."""
                return aux

        # Check if step_schedule is callable that accepts an integer
        # and returns a float
        try:
            if not isinstance(step_schd(1), float):
                raise ValueError("step_schedule must return a float")
        except Exception as e:
            raise ValueError(
                "step_schedule must accept an integer argument"
            ) from e

        self._step_schedule: StepSchedulerFn = step_schd

    @property
    def batch_size(self) -> BatchSizeFn:
        """The batch size schedule for the algorithm."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: BatchSizeArg) -> None:
        # Check if batch_size is callable or an integer
        if not callable(batch_size) and not isinstance(batch_size, int):
            raise TypeError("batch_size must be a callable or an int")

        # If batch_size is an integer, convert it to a callable
        if isinstance(batch_size, int):
            # Check if batch_size is positive
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")

            aux: int = batch_size

            # noinspection PyUnusedLocal
            def batch_size(n: int) -> int:
                """Return the batch size."""
                return aux

        # Check if batch_size is callable that accepts an integer and
        # returns an integer
        try:
            if not isinstance(batch_size(0), int):
                raise ValueError("batch_size must return an integer")
        except Exception as e:
            raise ValueError(
                "batch_size must accept an integer argument"
            ) from e

        self._batch_size: BatchSizeFn = batch_size


class DetentionParameters:
    """
    This class contains the detention parameters for the algorithm.
    """

    def __init__(
        self,
        tol: float = 1e-8,
        max_iter: int = 100_000,
        max_time: float = float("inf")
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
        if not (isinstance(max_iter, num.Integral)
                or max_iter == float("inf")):
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
            max_time = self.max_time
            time_fmt = str(timedelta(seconds=max_time))[:-4]
        max_iter_fmt = f"{self.max_iter:_}" if self.max_iter != float(
            "inf") else "∞"

        return (f"DetentionParameters(tol={self.tol:.2e}, "
                f"max_iter={max_iter_fmt}, max_time={time_fmt})")


class IterationParameters(Iterator[int]):
    r"""
    This class contains the iteration parameters for the algorithm.

    Examples
    --------
    Using this class is simple. You can initialize the class with the
    detention parameters and then iterate over the class to get the
    iteration number. Here is an example:

    >>> det_params = DetentionParameters(max_iter=3)
    >>> iter_params = IterationParameters(det_params)
    >>> for k in iter_params:
    ...     print(k)
    0
    1
    2

    The code above is equivalent to the following code:

    >>> det_params = DetentionParameters(max_iter=3)
    >>> iter_params = IterationParameters(det_params)
    >>> iter_params.init_params()
    >>> iter_params.update_iteration()
    >>> while not iter_params.detention_criteria():
    ...     iter_params.start_iteration()
    ...     print(iter_params.k)
    ...     iter_params.update_iteration()
    0
    1
    2
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

    @property
    def total_time(self) -> float:
        """
        The total time of the algorithm.

        :return: The total time of the algorithm.
        """
        return self.toc - self.tic

    def init_params(self) -> None:
        """
        Initializes the iteration metrics.

        This method sets the initial values for the iteration metrics
        used in the class.
        """
        self.k = -1
        self.tic = time.time()
        self.tic_ = time.time()
        self.toc = time.time()
        self.diff_t = 0
        self.w_dist = float("inf")

    def start_iteration(self) -> None:
        """
        Start the iteration.

        This method starts the iteration and sets the start time of the
        iteration to the current time.
        """
        self.tic_ = time.time()

    def update_iteration(self) -> None:
        """
        Update the iteration metrics.

        This method updates the iteration metrics used in the class.
        It updates the following attributes:
        - k: The iteration number (incremented by 1).
        - tic_: The start time of the iteration (set to the current time).
        - toc: The end time of the iteration (set to the current time).
        - diff_t: The time difference between tic_ and toc.

        :return: None
        """
        self.k += 1
        self.toc = time.time()
        self.diff_t = self.toc - self.tic_

    def update_wass_dist(self, wass_dist: float) -> float:
        """
        Update the Wasserstein distance.

        :param wass_dist: The Wasserstein distance.
        :return: The Wasserstein distance.
        """
        self.w_dist = wass_dist
        return wass_dist

    def detention_criteria(self) -> bool:
        """
        Determines the detention criteria for the algorithm.

        :return: True if the detention criteria is met, False otherwise.
        """
        return (
            # Reaches maximum iteration
            self.k >= self.det_params.max_iter
            # Reaches maximum time
            or self.total_time >= self.det_params.max_time
            # Achieves convergence in distance
            or self.w_dist < self.det_params.tol
        )

    def __repr__(self) -> str:
        w_dist_fmt = f"{self.w_dist:.6f}" if self.w_dist != float(
            "inf") else "∞"
        time_fmt = str(timedelta(seconds=self.total_time))[:-4]
        return (f"IterationParameters(k={self.k:_}, w_dist={w_dist_fmt}, "
                f"t={time_fmt}, Δt={self.diff_t * 1000:.2f} [ms])")

    def __iter__(self) -> Iterator[int]:
        self.init_params()
        return self

    def __next__(self) -> int:
        # At the beginning of the iteration, update the iteration metrics
        self.update_iteration()

        # If the detention criteria is met, raise StopIteration
        if self.detention_criteria():
            raise StopIteration

        # Start the timer for the next iteration
        self.start_iteration()

        # Return the iteration number
        return self.k
