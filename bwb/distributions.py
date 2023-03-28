from __future__ import annotations

import abc
import logging
import math
import time
import warnings
from collections import Counter
from itertools import product
from typing import TypeVar, Generic, Union, Any, Optional

import PIL.Image
import numpy as np
from PIL.Image import Image
from numpy import ma
from numpy.random import Generator

# Get logger
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)

# Configure a formatter
_formatter = logging.Formatter("%(asctime)s: %(levelname)s [%(name)s:%(lineno)s]\n> %(message)s")

# Configure a file handler
_file_handler = logging.FileHandler(f"logging.log")
_file_handler.setFormatter(_formatter)
# _log.addHandler(_file_handler)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
_log.addHandler(_stream_handler)

TSample = TypeVar("TSample")
GeneratorLike = Union[None, int, Generator]
Coord = tuple[int, int]


def default_likelihood(data: list[Coord]):
    ...


class RvsDistribution(abc.ABC, Generic[TSample]):
    """
    This class provides the interface for those distributions that can be sampled from them.
    """

    def __init__(self, seed: GeneratorLike):
        # Generator
        self.rng: np.random.Generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def rvs(self, *args, **kwargs) -> list[TSample]:
        """Random variates."""
        ...

    @abc.abstractmethod
    def draw(self, *args, **kwargs) -> TSample:
        """To draw a sample from the distribution."""
        ...

    def obtain_rng(self, random_state: GeneratorLike) -> Generator:
        """Obtain a random state. If the random state is `None`, returns the generator of the class"""
        return self.rng if random_state is None else np.random.default_rng(random_state)


class Distribution(RvsDistribution, abc.ABC, Generic[TSample]):
    @abc.abstractmethod
    def pdf(self, x: TSample | list[TSample], *args, **kwargs) -> float | list[float]:
        """Probability density function."""
        ...


class DistributionDraw(Distribution[Coord]):
    Coord = tuple[int, int]

    def __init__(self, image: Image, seed: GeneratorLike = None):
        """
        Initializer

        Parameters
        ----------
        image: Image
            The image to obtain the distribution.
        seed: int or generator
            The seed to create a generator and obtain replicable results.
        """
        # Generator
        super().__init__(seed)

        # Image and pdf
        self._image: Image = image
        self._matrix: np.ndarray = None
        self._log_matrix: np.ndarray = None
        self._support: list[Coord, ...] = None

        # Indices
        self._indices: list[Coord, ...] = None
        self._indices_inv: dict[Coord, int] = None

    @classmethod
    def fromarray(
            cls,
            array: np.ndarray,
            seed: GeneratorLike = None,
            ceil: int = 0
    ) -> DistributionDraw[Coord]:
        """
        Alternative initializer.

        Parameters
        ----------
        array: np.ndarray
            An array of integers with minimum value 0 and maximum value 255.
        seed: int or generator
            The seed to create a generator and obtain replicable results.
        """
        if ceil != 0:
            array = np.minimum(array, (255 - ceil) * np.ones_like(array), dtype=array.dtype)
        image = PIL.Image.fromarray(array)
        return cls(image=image, seed=seed)

    def _repr_png_(self):
        """
        iPython display hook support.

        Returns
        -------
            png version of the image as bytes
        """
        return self.image._repr_png_()

    @property
    def image(self) -> Image:
        """Original image."""
        return self._image

    @property
    def matrix(self) -> np.ndarray:
        """Matrix pdf obtained from the image."""
        if self._matrix is None:
            image_as_array = np.asarray(self.image)
            if len(image_as_array.shape) == 2:
                matrix = image_as_array
            else:
                matrix = image_as_array[:, :, 0]
            matrix = 1 - (matrix / 255)
            self._matrix = matrix / matrix.sum()
        return self._matrix

    @property
    def log_matrix(self) -> np.ndarray:
        if self._log_matrix is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._log_matrix = np.log(self.matrix)
        return self._log_matrix

    @property
    def shape(self) -> tuple:
        """The shape of the matrix"""
        return self.matrix.shape

    @property
    def indices(self) -> list[Coord, ...]:
        """A list of coordinates of the domain."""
        if self._indices is None:
            n, m = self.shape
            self._indices = [(i, j) for i, j in product(range(n), range(m))]
        return self._indices

    @property
    def indices_inv(self) -> dict[Coord, int]:
        """The inverse function that return the index if the component."""
        if self._indices_inv is None:
            self._indices_inv: dict[Coord, int] = {val: k for k, val in enumerate(self.indices)}
        return self._indices_inv

    @property
    def support(self) -> list[Coord, ...]:
        """Support of the distribution."""
        if self._support is None:
            non_zero_coord: np.ndarray = np.array(np.nonzero(self.matrix)).T
            self._support: list[Coord, ...] = [tuple(row) for row in non_zero_coord]
        return self._support

    @property
    def weights(self):
        """Return its correspondient weights in the same order of it support."""
        return self.pdf(self.support)

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> list[Coord]:
        # Set a random state
        rng = self.obtain_rng(random_state)
        # Sample with respect the weight matrix.
        samples = rng.choice(a=len(self.indices), size=size, p=self.matrix.flatten())
        return [self.indices[s] for s in samples]

    def draw(self, random_state: GeneratorLike = None) -> Coord:
        return self.rvs(size=1, random_state=random_state)[0]

    def pdf(self, x: Coord | list[Coord], **kwargs) -> float | list[float]:
        if isinstance(x, tuple):
            return self.matrix[x]
        indices_from_coord = [self.indices_inv[tup] for tup in x]
        return self.matrix.take(indices_from_coord)


class DistributionDrawBuilder:
    """
    Builder for the class DistributionDraw.
    """

    def __init__(self, seed: GeneratorLike = None, ceil: int = 0):
        self._rng: Generator = np.random.default_rng(seed)
        self._ceil: int = ceil

        self._dict_kwargs: dict[str, Any] = dict(seed=self._rng, ceil=self._ceil)

    def create(self, image: Image) -> DistributionDraw:
        return DistributionDraw(image=image, **self._dict_kwargs)

    def create_fromarray(self, array: np.ndarray) -> DistributionDraw:
        return DistributionDraw.fromarray(array=array, **self._dict_kwargs)

    def set_rng(self, rng: GeneratorLike):
        self._rng = np.random.default_rng(rng)
        self._dict_kwargs["seed"] = self._rng
        return self

    def set_ceil(self, ceil: int):
        self._ceil = ceil
        self._dict_kwargs["ceil"] = self._ceil
        return self


# noinspection PyUnboundLocalVariable
class PosteriorPiN(RvsDistribution[TSample], abc.ABC):
    """Abstract class used as protocol for the distributions used as Posterior."""

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[DistributionDraw],
            seed: GeneratorLike,
    ):
        super().__init__(seed)

        # Data = {x_i}_{i=1}^n
        self.data: list[TSample] = data

        self.shape = models[0].shape
        matrix_data = np.zeros(self.shape).astype(int)
        for coord in self.data:
            matrix_data[tuple(coord)] += 1
        matrix_data = ma.masked_less(matrix_data, 1)
        self._mask = ~matrix_data.mask
        self._counter_data = matrix_data[self._mask]

        # Models space M = {m_i}_{i=1}^M
        self.models: list[Distribution] = models
        self.models_index: np.ndarray[int] = np.arange(self.n_models)

        # Likelihood cache
        self._log_likelihood_cache: np.ndarray = None
        self._likelihood_cache: np.ndarray = None
        self.samples_counter: Counter = Counter()

        self.total_time = 0.

        _log.info(f"init PosteriorPiN: n_data={self.n_data}, n_models={self.n_models}")

    def __repr__(self):
        return self.__class__.__name__ + f"(n_data={self.n_data}, n_models={self.n_models})"

    @property
    def n_data(self) -> int:
        return len(self.data)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def log_likelihood_cache(self) -> np.ndarray[float]:
        """Likelihood cache to avoid to compute every time a same distribution, in order to be the same order of
        the list `models`."""
        if self._log_likelihood_cache is None or len(self._log_likelihood_cache) != self.n_models:
            _log.debug(f"Calculating the likelihood cache...")
            self._log_likelihood_cache = np.array([self.log_likelihood(m) for m in self.models])
            _log.debug(f"Likelihood cache calculated.")

            if sum(self._log_likelihood_cache > 0) == 1:
                warnings.warn("The likelihood support has only one element.")
        return self._log_likelihood_cache

    @property
    def likelihood_cache(self):
        if self._likelihood_cache is None:
            self._likelihood_cache = np.exp(self.log_likelihood_cache)
        return self._likelihood_cache

    def log_likelihood(self, m: DistributionDraw) -> float:
        return np.sum(self._counter_data * m.log_matrix[self._mask])

    def likelihood(self, m: Distribution) -> float:
        """
        Likelihood of the function:

        .. math:: \mathcal{L}_n(m) = \prod_{j=1}^{n} f_m(x_j)

        Parameters
        ----------
        m: Distribution
            The distribution to calculate the likelihood.

        Returns
        -------
        float
            The likelihood computed.
        """
        return np.exp(self.log_likelihood(m))

    def draw(self, *args, **kwargs) -> TSample:
        tic = time.time()

        to_return, i = self._draw(*args, **kwargs)

        self.samples_counter[i] += 1

        toc = time.time()
        _log.info(
            "=" * 10 + f" n samples: {self.samples_counter.total()}, total time: {toc - tic:.4f} [seg] " + "=" * 10)

        self.total_time += toc - tic
        return to_return

    @abc.abstractmethod
    def _draw(self, *args, **kwargs) -> tuple[TSample, int]:
        """To use template pattern"""
        ...

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> list[Distribution]:
        # Set a random state
        rng = self.obtain_rng(random_state)
        return [self.draw(random_state=rng) for _ in range(size)]

    def most_common(self, n: Optional[int] = None) -> list[Distribution]:
        return [self.models[k] for k, _ in self.samples_counter.most_common(n)]


class ExplicitPosteriorPiN(PosteriorPiN):
    """
    The explicit posterior probability, without using MCMC.
    """

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[DistributionDraw],
            seed: GeneratorLike = None,
            lazy_init: bool = True,
    ):
        super().__init__(data, models, seed)
        self._probabilities_arr = None

        if not lazy_init:
            # Calcular las verosimilitudes caché
            _ = self.likelihood_cache

    @property
    def probabilities_arr(self):
        if self._probabilities_arr is None:
            self._probabilities_arr = self.likelihood_cache / np.sum(self.likelihood_cache)
        return self._probabilities_arr

    def _draw(self, *args, random_state=None, **kwargs) -> tuple[TSample, int]:
        rng = self.obtain_rng(random_state)
        i = rng.choice(a=self.models_index, p=self.probabilities_arr)
        return self.models[i], i

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> list[Distribution]:
        tic = time.time()

        # Set a random state
        rng = self.obtain_rng(random_state)
        index_list = rng.choice(a=self.models_index, size=size, p=self.probabilities_arr)

        self.samples_counter.update(index_list)

        toc = time.time()
        _log.info(
            "=" * 10 + f" n samples: {self.samples_counter.total()}, total time: {toc - tic:.4f} [seg] " + "=" * 10)
        self.total_time += toc - tic
        return [self.models[i] for i in index_list]


class MCMCPosteriorPiN(PosteriorPiN[TSample], abc.ABC):
    """Abstract class for the implementations of the posterior, using MCMC algorithms."""

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[Distribution],
            seed: GeneratorLike,
            lazy_init: bool,
            mixing_time,
            precision,
    ):
        super().__init__(data, models, seed)

        # Number of the iteration
        self._i = 0
        self.mixing_time = mixing_time
        self.precision = precision

        self.last_measure = None
        self.last_i: int = None
        self._lazy_init = lazy_init

        self.steps_counter: Counter = Counter()

    def __repr__(self):
        return (self.__class__.__name__
                + f"(n_data={self.n_data}, n_models={self.n_models}, "
                  f"n_samples={self.samples_counter.total()}, last_i={self.last_i})")

    @abc.abstractmethod
    def _first_step(self, rng: GeneratorLike):
        ...

    def _post_init_(self):
        if not self._lazy_init:
            self._first_step(self._first_step(self.rng))

    def _draw(self, *args, random_state=None, **kwargs) -> tuple[TSample, int]:

        # Calcular el t_mix(eps)
        t_mix_eps = int(math.log2(1 / self.precision) * self.mixing_time + 1)
        _log.debug(f"t_mix(eps) = {t_mix_eps} steps")

        # Dejar pasar algunas iteraciones para obtener resultados indep
        for _ in range(t_mix_eps - 1):
            self.step(random_state)

        return self.step(random_state)

    @abc.abstractmethod
    def _step(self, random_state: GeneratorLike) -> tuple[Distribution, int]:
        """To use template pattern. Returns a distribution and its index."""
        ...

    def step(self, random_state: GeneratorLike = None) -> tuple[DistributionDraw, int]:
        _log.debug("=" * 10 + f" mcmc iter = {self._i} " + "=" * 10)
        self._i += 1
        step, i = self._step(random_state=random_state)
        self.steps_counter[i] += 1
        return step, i


class MetropolisPosteriorPiN(MCMCPosteriorPiN[TSample]):

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[Distribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
            mixing_time: int = 150,
            precision: float = 1e-2,
    ):
        super().__init__(data, models, seed, lazy_init, mixing_time, precision)
        probabilities: np.ndarray[float] = np.isfinite(self.log_likelihood_cache).astype(float)
        probabilities /= probabilities.sum()
        self.probabilities = probabilities
        self._post_init_()

    def _first_step(self, rng: GeneratorLike):
        if self.last_measure is None:
            _log.info("Executing _first_step")
            # Choose a model
            self.last_i = int(rng.choice(self.n_models, p=self.probabilities.copy()))
            _log.info(f"First model selected: {self.last_i}.")
            self.last_measure = self.models[self.last_i]

    def _step(self, random_state: GeneratorLike = None) -> tuple[Distribution, int]:
        # Set generator
        rng = self.obtain_rng(random_state)

        # If this is the first time that executes this instruction
        self._first_step(rng)
        last_i = self.last_i

        if sum(np.isfinite(self.log_likelihood_cache)) == 1:
            return self.last_measure, self.last_i

        # Draw a candidate
        probabilities = self.probabilities.copy()
        probabilities[last_i] = 0.  # Remove the last distribution
        probabilities /= probabilities.sum()

        next_i = int(rng.choice(self.models_index, p=probabilities))

        # Compute the acceptance probability
        log_likelihood_diff = self.log_likelihood_cache[next_i] - self.log_likelihood_cache[last_i]
        _log.debug(f"{log_likelihood_diff = }")

        # Acceptance / Rejection
        if (log_likelihood_diff >= 1
                or rng.uniform(low=0, high=1) < np.exp(log_likelihood_diff)):
            _log.info(f"i = {self._i}: {last_i = } -> {next_i = }.")
            # Add to the history
            mu_star = self.models[next_i]
            self.last_measure = mu_star

            # Update the last_i
            self.last_i = next_i

        return self.last_measure, last_i


class AlternativeMetropolisPosteriorPiN(MetropolisPosteriorPiN[TSample]):
    def __init__(
            self,
            data: list[TSample],
            models: list[Distribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
            mixing_time: int = 150,
            precision: float = 1e-2,
    ):
        super().__init__(data, models, seed, lazy_init, mixing_time, precision)
        self.models = [m for m in self.models if self.likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.rng)


class GibbsPosteriorPiN(MCMCPosteriorPiN[TSample]):

    def __init__(
            self,
            data: list[TSample],
            models: list[Distribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
            mixing_time: int = 150,
            precision: float = 1e-2,
    ):
        super().__init__(data, models, seed, lazy_init, mixing_time, precision)
        self.possible_models = np.arange(self.n_models)
        self.probs_cache: dict[int, np.ndarray] = dict()
        self.likelihood_sum: float = 0.

        if not lazy_init:
            self._first_step(self.rng)

    def __repr__(self):
        return (self.__class__.__name__
                + f"(n_data={self.n_data}, n_models={self.n_models}, "
                  f"n_samples={self.samples_counter.total()}, last_i={self.last_i},"
                  f" likelihood_sum={self.likelihood_sum:.4e})")

    def _first_step(self, rng):
        if self.last_measure is None:
            _log.info("Executing _first_step")
            self.likelihood_sum = np.sum(self.likelihood_cache)
            _log.debug(f"likelihood_sum = {self.likelihood_sum}")
            # Choose a model
            possible_distributions = np.arange(self.n_models)[np.isfinite(self.log_likelihood_cache)]
            self.last_i = int(rng.choice(possible_distributions))
            self.last_measure = self.models[self.last_i]
            _log.info(f"First model selected: {self.last_i}.")

    def _step(self, random_state=None) -> tuple[Distribution, int]:
        # Set generator
        rng = self.obtain_rng(random_state)

        # If this is the first time that executes this instruction
        if self.last_measure is None:
            self._first_step(rng)

        last_i = self.last_i

        if sum(np.isfinite(self.log_likelihood_cache)) == 1:
            return self.last_measure, self.last_i

        # Draw a candidate
        probs = self.likelihood_cache.copy()
        probs[last_i] = 0.
        probs /= probs.sum()

        next_i = int(rng.choice(self.possible_models, p=probs))

        # Compute the acceptance probability
        acceptance_prob = ((self.likelihood_sum - self.likelihood_cache[next_i])
                           / (self.likelihood_sum - self.likelihood_cache[last_i]))
        _log.debug(f"{acceptance_prob = }")

        # Acceptance / Rejection
        if (acceptance_prob >= 1
                or rng.uniform(low=0, high=1) < acceptance_prob):
            _log.info(f"i = {self._i}: {last_i = } -> {next_i = }.")

            # Add to the history
            mu_star = self.models[next_i]
            self.last_measure = mu_star

            # Update the last_i
            self.last_i = next_i

        return self.last_measure, last_i


class AlternativeGibbsPosteriorPiN(GibbsPosteriorPiN[TSample]):
    def __init__(
            self,
            data: list[TSample],
            models: list[Distribution],
            seed=None,
            lazy_init=False,
            mixing_time: int = 150,
            precision: float = 1e-2,
    ):
        super().__init__(data, models, seed, lazy_init, mixing_time, precision)

        self.models = [m for m in self.models if self.log_likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.rng)
