import numbers as num
import time
import typing as t
from copy import deepcopy
from datetime import timedelta
from typing import Iterator


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
            max_time = self.max_time
            time_fmt = str(timedelta(seconds=max_time))[:-4]
        max_iter_fmt = f"{self.max_iter:_}" if self.max_iter != float("inf") else "∞"

        return f"DetentionParameters(tol={self.tol:.2e}, max_iter={max_iter_fmt}, max_time={time_fmt})"


class IterationParameters(Iterator[int]):
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

    @property
    def total_time(self) -> float:
        return self.toc - self.tic

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
        self.k = -1
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

        :return: None
        """
        self.k += 1
        self.toc = time.time()
        self.diff_t = self.toc - self.tic_

    def update_wass_dist(self, wass_dist):
        self.w_dist = wass_dist
        return wass_dist

    def detention_criteria(self) -> bool:
        """
        Determines the detention criteria for the algorithm.

        :return: True if the detention criteria is met, False otherwise.
        """
        return (
            self.k >= self.det_params.max_iter  # Reaches maximum iteration
            or self.total_time >= self.det_params.max_time  # Reaches maximum time
            or self.w_dist < self.det_params.tol  # Achieves convergence in distance
        )

    def __repr__(self) -> str:
        w_dist_fmt = f"{self.w_dist:.6f}" if self.w_dist != float("inf") else "∞"
        time_fmt = str(timedelta(seconds=self.total_time))[:-4]
        return f"IterationParameters(k={self.k:_}, w_dist={w_dist_fmt}, t={time_fmt}, Δt={self.diff_t * 1000:.2f} [ms])"

    def __iter__(self):
        self.init_params()
        return self

    def __next__(self) -> int:
        # At the beginning of the iteration, update the iteration metrics
        self.update_iteration()

        # If the detention criteria is met, raise StopIteration
        if self.detention_criteria():
            raise StopIteration

        # Start the timer for the iteration
        self.start_iteration()

        # Return the iteration number
        return self.k


class HistoryOptions(t.TypedDict, total=False):
    pos_wgt: bool
    distr: bool
    pos_wgt_samp: bool
    distr_samp: bool


HistoryOptionsLiteral = t.Literal["pos_wgt", "distr", "pos_wgt_samp", "distr_samp"]


class History[DistributionT, pos_wgt_t]:
    """
    This class contains the history logic of the algorithm.
    """

    HISTORY_OPTIONS: HistoryOptions = {
        "pos_wgt": False,
        "distr": False,
        "pos_wgt_samp": False,
        "distr_samp": False,
    }

    def __init__(
        self,
        pos_wgt: bool = False,
        distr: bool = False,
        pos_wgt_samp: bool = False,
        distr_samp: bool = False,
    ):
        # Dictionary by default
        self.options: HistoryOptions = deepcopy(self.HISTORY_OPTIONS)

        # Update the dictionary
        self.options["pos_wgt"] = pos_wgt
        self.options["distr"] = distr
        self.options["pos_wgt_samp"] = pos_wgt_samp
        self.options["distr_samp"] = distr_samp

        # Initialize the histories
        self._create_distribution: t.Optional[t.Callable[[pos_wgt_t], DistributionT]] = (
            None
        )
        self._get_pos_wgt_from_dist: t.Optional[
            t.Callable[[DistributionT], pos_wgt_t]
        ] = None
        self.pos_wgt: list[pos_wgt_t] = []
        self.distr: list[DistributionT] = []
        self.pos_wgt_samp: list[list[pos_wgt_t]] = []
        self.distr_samp: list[list[DistributionT]] = []

    def add_key_value(self, key: HistoryOptionsLiteral, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"Value must be a boolean, not {type(value)}")
        if key not in self.HISTORY_OPTIONS:
            raise KeyError(f"Invalid key: {key}")
        self.options[key] = value

    @property
    def create_distr(self) -> t.Callable[[pos_wgt_t], DistributionT]:
        """The function to create a distribution from the position and weight."""
        if self._create_distribution is None:
            raise ValueError("create_distr must be set")
        return self._create_distribution

    @create_distr.setter
    def create_distr(self, create_distribution):
        if not callable(create_distribution):
            raise TypeError("create_distribution must be a callable")
        self._create_distribution: t.Callable[[pos_wgt_t], DistributionT] = (
            create_distribution
        )

    @property
    def get_pos_wgt_from_dist(self) -> t.Callable[[DistributionT], pos_wgt_t]:
        """The function to get the position and weight from the distribution."""
        if self._get_pos_wgt_from_dist is None:
            raise ValueError("get_pos_wgt_from_dist must be set")
        return self._get_pos_wgt_from_dist

    @get_pos_wgt_from_dist.setter
    def get_pos_wgt_from_dist(self, get_pos_wgt_from_dist):
        if not callable(get_pos_wgt_from_dist):
            raise TypeError("get_pos_wgt_from_dist must be a callable")
        self._get_pos_wgt_from_dist: t.Callable[[DistributionT], pos_wgt_t] = (
            get_pos_wgt_from_dist
        )

    def set_params(
        self,
        pos_wgt: bool = None,
        distr: bool = None,
        pos_wgt_samp: bool = None,
        distr_samp: bool = None,
        create_distribution: t.Callable[[pos_wgt_t], DistributionT] = None,
        get_pos_wgt_from_dist: t.Callable[[DistributionT], pos_wgt_t] = None,
    ):
        self.options["pos_wgt"] = (
            pos_wgt if pos_wgt is not None else self.options["pos_wgt"]
        )
        self.options["distr"] = distr if distr is not None else self.options["distr"]
        self.options["pos_wgt_samp"] = (
            pos_wgt_samp if pos_wgt_samp is not None else self.options["pos_wgt_samp"]
        )
        self.options["distr_samp"] = (
            distr_samp if distr_samp is not None else self.options["distr_samp"]
        )

        self.create_distr = (
            create_distribution
            if create_distribution is not None
            else self.create_distr
        )
        self.get_pos_wgt_from_dist = (
            get_pos_wgt_from_dist
            if get_pos_wgt_from_dist is not None
            else self.get_pos_wgt_from_dist
        )

    def init_histories(self, lst_mu_0: t.Sequence[DistributionT], pos_wgt_0: pos_wgt_t):
        """
        Initialize the histories for the algorithm.

        This method should initialize the histories for the algorithm.

        :param lst_mu_0: The list of distributions sampled by the sampler at the first iteration.
        :param pos_wgt_0: The position and weight that come from the first sample.
        """
        if self.options["pos_wgt"]:
            self.pos_wgt = [pos_wgt_0]
        if self.options["distr"]:
            self.distr = [self.create_distr(pos_wgt_0)]
        if self.options["pos_wgt_samp"]:
            self.pos_wgt_samp = [[self.get_pos_wgt_from_dist(mu) for mu in lst_mu_0]]
        if self.options["distr_samp"]:
            self.distr_samp = [lst_mu_0]

    def update_pos_wgt(self, pos_wgt_kp1):
        """
        Update the position and weight for the next iteration.

        This method should update the position and weight for the next iteration.

        :param pos_wgt_kp1: The position and weight that come from the next sample.
        """
        if self.options["pos_wgt"]:
            self.pos_wgt.append(pos_wgt_kp1)

    def update_distr(self, pos_wgt_kp1):
        """
        Update the distribution history for the next iteration.

        This method should update the distribution history for the next iteration.

        :param pos_wgt_kp1: The position and weight that come from the next sample.
        """
        if self.options["distr"]:
            self.distr.append(self.create_distr(pos_wgt_kp1))

    def update_pos_wgt_samp(self, lst_mu_k):
        """
        Update the position and weight sampler history for the next iteration.

        This method should update the position and weight sampler history for the next iteration.

        :param lst_mu_k: The list of distributions sampled by the sampler at the current iteration.
        """
        if self.options["pos_wgt_samp"]:
            self.pos_wgt_samp.append(
                [self.get_pos_wgt_from_dist(mu) for mu in lst_mu_k]
            )

    def update_distr_samp(self, lst_mu_k):
        """
        Update the distribution sampler history for the next iteration.

        This method should update the distribution sampler history for the next iteration.

        :param lst_mu_k: The list of distributions sampled by the sampler at the current iteration.
        """
        if self.options["distr_samp"]:
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
        self.update_pos_wgt_samp(lst_mu_k)
        self.update_distr_samp(lst_mu_k)

    def has_pos_wgt(self) -> bool:
        return self.options["pos_wgt"]

    def has_distr(self) -> bool:
        return self.options["distr"]

    def has_pos_wgt_samp(self) -> bool:
        return self.options["pos_wgt_samp"]

    def has_distr_samp(self) -> bool:
        return self.options["distr_samp"]

    def has_histories(self):
        return any(
            [
                self.has_pos_wgt(),
                self.has_distr(),
                self.has_pos_wgt_samp(),
                self.has_distr_samp(),
            ]
        )

    def __getitem__(self, key: HistoryOptionsLiteral):
        match key:
            case "pos_wgt":
                return self.pos_wgt
            case "distr":
                return self.distr
            case "pos_wgt_samp":
                return self.pos_wgt_samp
            case "distr_samp":
                return self.distr_samp
            case _:
                raise KeyError(f"Invalid key: {key}")

    def __repr__(self) -> str:
        to_return = f"{self.__class__.__name__}("
        for key, value in self.options.items():
            to_return += f"{key}={value}, "
        to_return += f"len={len(self)})"
        return to_return

    def __len__(self) -> int:
        if self.has_pos_wgt():
            return len(self.pos_wgt)
        if self.has_distr():
            return len(self.distr)
        if self.has_pos_wgt_samp():
            return len(self.pos_wgt_samp)
        if self.has_distr_samp():
            return len(self.distr_samp)
        return 0


class ReportOptions(t.TypedDict, total=False):
    iter: bool
    w_dist: bool
    step_schd: bool
    total_time: bool
    dt: bool
    dt_per_iter: bool


ReportOptionsLiteral = t.Literal[
    "iter", "w_dist", "step_schd", "total_time", "dt", "dt_per_iter"
]


class Report:
    """
    This class contains the report logic of the algorithm.
    """

    INCLUDE_OPTIONS: ReportOptions = {
        "iter": True,
        "w_dist": False,
        "step_schd": True,
        "total_time": True,
        "dt": False,
        "dt_per_iter": True,
    }

    def __init__(
        self,
        iter_params: IterationParameters,
        report_every: int = 10,
        include_dict: ReportOptions = None,
        len_bar: int = 5,
    ):
        self.iter_params = iter_params
        self.report_every = report_every
        self.len_bar = len_bar

        # Dictionary by default
        self.include: ReportOptions = deepcopy(self.INCLUDE_OPTIONS)

        # Update the dictionary
        include_dict: ReportOptions = include_dict or {}
        for key, value in include_dict.items():
            self.add_key_value(key, value)

    def add_key_value(self, key: ReportOptionsLiteral, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f"Value must be a boolean, not {type(value)}")
        if key not in self.include:
            raise KeyError(f"Invalid key: {key}")
        self.include[key] = value

    def set_params(
        self,
        include_dict: ReportOptions = None,
        len_bar: int = None,
    ):
        include_dict = include_dict or {}
        for key, value in include_dict.items():
            self.add_key_value(key, value)

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

        if self.include["iter"]:
            report += f"k = {self.iter_params.k}, "

        if self.include["w_dist"]:
            report += f"Wass. dist. = {self.iter_params.w_dist:.6f}, "

        if self.include["step_schd"]:
            report += f"gamma_k = {gamma_k:.2%}, "

        if self.include["total_time"]:
            total_time = self.iter_params.total_time
            time_fmt = str(timedelta(seconds=total_time))[:-4]
            report += f"t = {time_fmt}, "

        if self.include["dt"]:
            report += f"Δt = {(self.iter_params.diff_t) * 1000:.2f} [ms], "

        if self.include["dt_per_iter"]:
            report += f"Δt per iter. = {(self.iter_params.total_time) * 1000 / (self.iter_params.k + 1):.2f} [ms/iter], "

        report = report[:-2] + " " + bar

        return report

    def is_report_iter(self) -> bool:
        return self.iter_params.k % self.report_every == 0
