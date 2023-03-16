from __future__ import annotations

import abc
import logging
import warnings
from collections import Counter
from itertools import product
from typing import TypeVar, Generic, Union, Any, Optional, Protocol, Iterable

import PIL.Image
import numpy as np
from PIL.Image import Image
from numpy._typing import ArrayLike
from numpy.random import Generator

# Get logger

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

# Configure a formatter
_formatter = logging.Formatter("%(asctime)s: %(levelname)s [%(name)s:%(lineno)s] %(message)s")

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


class HasGenerator(abc.ABC):
    """For classes that has a generator"""

    def __init__(self, seed: GeneratorLike):
        # Generator
        self.random_state: np.random.Generator = np.random.default_rng(seed)

    def get_rng(self, random_state: GeneratorLike) -> Generator:
        """Obtain a random state. If the random state is `None`, returns the generator of the class"""
        return self.random_state if random_state is None else np.random.default_rng(random_state)


class RvsDistribution(Generic[TSample], Protocol):
    """
    This class provides the interface for those distributions that can be sampled from them.
    """

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> np.ndarray[TSample]:
        """Random variates."""
        ...

    def draw(self, random_state: GeneratorLike = None) -> TSample:
        """To draw a sample from the distribution."""
        ...


class DiscreteDistribution(RvsDistribution[TSample], HasGenerator, Generic[TSample]):
    """
    Class to represent a finite supported discrete distribution probability.
    """

    def __init__(
            self,
            weights: ArrayLike[float],
            support: ArrayLike[TSample],
            seed: GeneratorLike = None
    ):
        super().__init__(seed=seed)

        self.weights: np.ndarray = np.asarray(weights)
        self.support: np.ndarray = np.asarray(support)

        if len(self.weights) != len(self.support):
            raise ValueError("The arrays must have the same length.")

        if self.weights.sum() != 1:
            raise ValueError("The weights must sum 1.0")

        # Simplified hidden support and hidden weights
        self._support = np.array(list(set(self.support)))
        self._weights = np.array([
            np.sum(self.weights[self.support == x]) for x in self._support
        ])

    def pdf(self, x: TSample | ArrayLike[TSample], *args, **kwargs) -> float | ArrayLike[float]:
        """Probability density function."""

        if not isinstance(x, Iterable):
            return self._weights[np.where(self._support == x)]
        return np.array([self.pdf(x_) for x_ in x])
        # return np.take(self._weights)

    # def log_pdf(self: x: TSample | ArrayLike[TSample], *args, **kwargs) -> :

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> np.ndarray[TSample]:
        rng = self.get_rng(random_state)
        return rng.choice(a=self.support, size=size, p=self.weights)

    def draw(self, random_state: GeneratorLike = None) -> TSample:
        return self.rvs(size=1, random_state=random_state)[0]


# TODO: Cambiar el inicializador para ser construido a partir de arreglos normalizados en vez de imagenes
# TODO:
class DistributionDraw(DiscreteDistribution[int]):
    # Coord = tuple[int, int]

    def __init__(
            self,
            weights: ArrayLike[float],
            support: ArrayLike[int],
            shape: tuple[int, int],
            seed: GeneratorLike = None
    ):
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
        super().__init__(weights=weights, support=support, seed=seed)

        # Image and probability matrix
        self._image: Image = None
        self._matrix: np.ndarray = None
        self.shape: tuple = shape

        n, m = shape
        self.ind_to_coord = [(i, j) for i, j in product(range(n), range(m))]
        self.coord_to_ind = np.arange(n * m).reshape(shape)

    @classmethod
    def from_array(
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
        # Make the array as a probability distribution
        array: np.ndarray = 1 - array / 255
        array /= array.sum()
        # Obtain the support and the weights
        N, M = shape = array.shape
        non_zero_supp = np.array(np.nonzero(array))
        supp = non_zero_supp[1] + non_zero_supp[0] * M
        weights = array.take(supp)
        # Create the insance and set the probability matrix
        instance = cls(support=supp, weights=weights, shape=shape, seed=seed)
        instance._matrix = array
        instance._image = image
        return instance

    @classmethod
    def from_image(cls, image: Image, seed: GeneratorLike = None):
        image_as_array = np.asarray(image)
        if len(image_as_array.shape) == 2:
            matrix = image_as_array
        else:
            matrix = image_as_array[:, :, 0]
        instance = cls.from_array(matrix, seed=seed)
        instance._image = image
        return instance

    @classmethod
    def from_weights(
            cls,
            weights: ArrayLike[float],
            support: ArrayLike[int],
            shape: tuple[int, int],
            seed: GeneratorLike = None
    ):
        return cls(weights, support, shape, seed)

    def _repr_png_(self):
        """
        iPython display hook support.

        Returns
        -------
            png version of the image as bytes
        """
        return self.image._repr_png_()

    @property
    def matrix(self) -> np.ndarray:
        """Matrix pdf obtained from the image."""
        if self._matrix is None:
            to_return = np.zeros(self.shape)
            for row, weight in zip(self.support.astype(int), self.weights):
                coord = self.ind_to_coord[row]
                to_return[coord] += weight
            self._matrix = to_return
        return self._matrix

    @property
    def image(self) -> Image:
        """Original image."""
        if self._image is None:
            matrix: np.ndarray = np.ceil(255 - 255 * self.matrix / self.matrix.max()).astype("uint8")
            self._image = PIL.Image.fromarray(matrix)
        return self._image

    # @property
    # def indices(self) -> list[Coord]:
    #     """A list of coordinates of the domain."""
    #     if self._indices is None:
    #         n, m = self.shape
    #         self._indices = [(i, j) for i, j in product(range(n), range(m))]
    #     return self._indices
    #
    # @property
    # def indices_inv(self) -> dict[Coord, int]:
    #     """The inverse function that return the index if the component."""
    #     if self._indices_inv is None:
    #         self._indices_inv: dict[Coord, int] = {val: k for k, val in enumerate(self.indices)}
    #     return self._indices_inv

    # @property
    # def support(self) -> list[Coord, ...]:
    #     """Support of the distribution."""
    #     if self._support is None:
    #         non_zero_coord: np.ndarray = np.array(np.nonzero(self.matrix)).T
    #         self._support: list[Coord, ...] = [tuple(row) for row in non_zero_coord]
    #     return self._support

    # @property
    # def weights(self):
    #     """Return its correspondient weights in the same order of it support."""
    #     return self.pdf(self.support)

    # def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> list[Coord]:
    #     # Set a random state
    #     random_state = self.get_rng(random_state)
    #     # Sample with respect the weight matrix.
    #     samples = random_state.choice(a=len(self.indices), size=size, p=self.matrix.flatten())
    #     return [self.indices[s] for s in samples]
    #
    # def draw(self, random_state: GeneratorLike = None) -> Coord:
    #     return self.rvs(size=1, random_state=random_state)[0]

    def pdf(self, x: int | list[int], **kwargs) -> float | list[float]:
        # if isinstance(x, tuple):
        #     return self.matrix[x]
        # indices_from_coord = [self.indices_inv[tup] for tup in x]
        return self.matrix.take(x)


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
        return DistributionDraw.from_array(array=array, **self._dict_kwargs)

    def set_rng(self, rng: GeneratorLike):
        self._rng = np.random.default_rng(rng)
        self._dict_kwargs["seed"] = self._rng
        return self

    def set_ceil(self, ceil: int):
        self._ceil = ceil
        self._dict_kwargs["ceil"] = self._ceil
        return self


# TODO: Hacerle un método fit en donde se ajusten los modelos y los datos, pero en el init se ajuste el generador y
#  la verosimilitud
class PosteriorPiN(HasGenerator, RvsDistribution[TSample], abc.ABC):
    """Abstract class used as protocol for the distributions used as Posterior."""

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[DiscreteDistribution],
            seed: GeneratorLike,
    ):
        super().__init__(seed)

        # Number of the iteration
        self._i = 0

        # Data = {x_i}_{i=1}^n
        self.data: list[TSample] = data
        # Models space M = {m_i}_{i=1}^M
        self.models: list[DiscreteDistribution] = models
        # Likelihood cache
        self._likelihood_cache: np.ndarray = None
        self.counter: Counter = Counter()

        _log.info(f"init PosteriorPiN: n_data={self.n_data}, n_models={self.n_models}")

    @property
    def n_data(self) -> int:
        return len(self.data)

    @property
    def n_models(self) -> int:
        return len(self.models)

    @property
    def likelihood_cache(self) -> np.ndarray[float]:
        """Likelihood cache to avoid to compute every time a same distribution, in order to be the same order of
        the list `models`."""
        if self._likelihood_cache is None or len(self._likelihood_cache) != self.n_models:
            _log.debug(f"Calculating the likelihood cache...")
            self._likelihood_cache = np.array([self.likelihood(m) for m in self.models])
            _log.debug(f"Likelihood cache calculated.")

            if sum(self._likelihood_cache > 0) == 1:
                warnings.warn("The likelihood support has only one element.")
        return self._likelihood_cache

    def __repr__(self):
        return self.__class__.__name__ + f"(n_data={self.n_data}, n_models={self.n_models})"

    def likelihood(self, m: DiscreteDistribution) -> float:
        """
        Likelihood of the function:

        .. math:: \mathcal{L}_n(m) = \prod_{j=1}^{n} f_m(x_j)

        Parameters
        ----------
        m: DiscreteDistribution
            The distribution to calculate the likelihood.

        Returns
        -------
        float
            The likelihood computed.
        """
        evaluations_prod = 1.
        # Iterate over every value of the data
        for i in range(self.n_data):
            evaluations_prod *= m.pdf(self.data[i])
            # If the evaluation is zero, return 0 directly
            if evaluations_prod == 0:
                return 0.

        # Return the product of evaluations
        return evaluations_prod

    def rvs(self, size: int = 1, random_state: GeneratorLike = None) -> list[DiscreteDistribution]:
        # Set a random state
        rng = self.get_rng(random_state)
        return [self.draw(random_state=rng) for _ in range(size)]

    def draw(self, random_state: GeneratorLike = None) -> DiscreteDistribution:
        _log.debug("=" * 10 + f" i = {self._i} " + "=" * 10)
        self._i += 1
        draw, i = self._draw(random_state=random_state)
        self.counter[i] += 1
        return draw

    @abc.abstractmethod
    def _draw(self, random_state: GeneratorLike) -> tuple[DiscreteDistribution, int]:
        """To use template pattern. Returns a distribution and its index."""
        ...

    def most_common(self, n: Optional[int] = None) -> list[DiscreteDistribution]:
        return [self.models[k] for k, _ in self.counter.most_common(n)]


class MCMCPosteriorPiN(PosteriorPiN[TSample], abc.ABC):
    """Abstract class for the implementations of the posterior, using MCMC algorithms."""

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[DiscreteDistribution],
            seed: GeneratorLike,
            lazy_init: bool,
    ):
        super().__init__(data, models, seed)
        # History {mu^(i)}_{i=1}^N
        self.history: list[DiscreteDistribution] = []
        self.last_i: int = None
        if not lazy_init:
            self._first_step(self.random_state)

    def __repr__(self):
        return (self.__class__.__name__
                + f"(n_data={self.n_data}, n_models={self.n_models}, "
                  f"n_samples={len(self.history) - 1}, last_i={self.last_i})")

    @abc.abstractmethod
    def _first_step(self, rng: GeneratorLike):
        ...


class MetropolisPosteriorPiN(MCMCPosteriorPiN[TSample]):

    def __init__(
            self,
            data: list[TSample, ...],
            models: list[DiscreteDistribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
    ):
        super().__init__(data, models, seed, lazy_init)
        self.possible_models: np.ndarray[int] = np.arange(self.n_models)

    def _first_step(self, rng):
        _log.info("Executing _first_step")
        # To avoid a measure with zero likelihood
        probabilities = (self.likelihood_cache > 0).astype(float)
        probabilities /= probabilities.sum()
        # Choose a model
        self.last_i = int(rng.choice(self.n_models, p=probabilities))
        _log.info(f"First model selected: {self.last_i}.")
        self.history.append(self.models[self.last_i])
        # Update the counter
        self.counter[self.last_i] += 1

    def _draw(self, random_state: GeneratorLike = None) -> tuple[DiscreteDistribution, int]:
        # Set generator
        rng = self.get_rng(random_state)

        # If this is the first time that executes this instruction
        if len(self.history) == 0:
            self._first_step(rng)
        last_i = self.last_i

        if sum(self.likelihood_cache > 0) == 1:
            mu_i = self.history[-1]
            self.history.append(mu_i)
            return self.history[-1], self.last_i

        # Draw a uniform
        u = rng.uniform(low=0, high=1)
        _log.debug(f"{u = }")

        # Draw a candidate
        probabilities = np.ones(self.n_models)
        probabilities[last_i] = 0.  # Remove the last distribution
        probabilities /= probabilities.sum()

        next_i = int(rng.choice(self.possible_models, p=probabilities))

        # Compute the acceptance probability
        A_mu_i_mu_star = min(
            1.,
            (self.likelihood_cache[next_i])
            / (self.likelihood_cache[last_i])
        )
        _log.debug(f"{A_mu_i_mu_star = }")
        _log.debug(f"{self.likelihood_cache[next_i] = }")

        # Acceptance / Rejection
        if u < A_mu_i_mu_star:
            _log.info(f"i = {self._i}: {last_i = } -> {next_i = }.")
            # Add to the history
            mu_star = self.models[next_i]
            self.history.append(mu_star)

            # Update the last_i
            self.last_i = next_i

        else:
            # Repeat the last sample
            mu_i = self.history[-1]
            self.history.append(mu_i)

        return self.history[-1], last_i


class AlternativeMetropolisPosteriorPiN(MetropolisPosteriorPiN[TSample]):
    def __init__(
            self,
            data: list[TSample],
            models: list[DiscreteDistribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
    ):
        super().__init__(data, models, seed, True)
        self.models = [m for m in self.models if self.likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.random_state)


class GibbsPosteriorPiN(MCMCPosteriorPiN[TSample]):

    def __init__(
            self,
            data: list[TSample],
            models: list[DiscreteDistribution],
            seed: GeneratorLike = None,
            lazy_init: bool = False,
    ):
        super().__init__(data, models, seed, True)
        self.possible_models = np.arange(self.n_models)
        self.probs_cache: dict[int, np.ndarray] = dict()
        self.likelihood_sum: float = 0.

        if not lazy_init:
            self._first_step(self.random_state)

    def __repr__(self):
        return (self.__class__.__name__
                + f"(n_data={self.n_data}, n_models={self.n_models}, "
                  f"n_samples={len(self.history) - 1}, last_i={self.last_i},"
                  f" likelihood_sum={self.likelihood_sum:.4e})")

    def _first_step(self, rng):
        _log.info("Executing _first_step")
        self.likelihood_sum = np.sum(self.likelihood_cache)
        _log.debug(f"likelihood_sum = {self.likelihood_sum}")
        # Choose a model
        possible_distributions = np.arange(self.n_models)[self.likelihood_cache > 0]
        self.last_i = int(rng.choice(possible_distributions))
        self.history.append(self.models[self.last_i])
        _log.info(f"First model selected: {self.last_i}.")
        # Update the counter
        self.counter[self.last_i] += 1

    def _draw(self, random_state=None) -> tuple[DiscreteDistribution, int]:
        # Set generator
        rng = self.get_rng(random_state)

        # If this is the first time that executes this instruction
        if len(self.history) == 0:
            self._first_step(rng)
        last_i = self.last_i

        if sum(self.likelihood_cache > 0) == 1:
            mu_i = self.history[-1]
            self.history.append(mu_i)
            return self.history[-1], self.last_i

        # Draw a uniform
        u = rng.uniform(low=0, high=1)
        _log.debug(f"{u = }")

        # Draw a candidate
        if last_i not in self.probs_cache:
            _log.debug(f"Computing the cache probabilities for the draw {last_i}.")
            self.probs_cache[last_i] = self.likelihood_cache.copy()
            self.probs_cache[last_i][last_i] = 0.
            self.probs_cache[last_i] /= self.probs_cache[last_i].sum()

        next_i = int(rng.choice(self.possible_models, p=self.probs_cache[last_i]))

        # Compute the acceptance probability
        A_mu_i_mu_star = min(
            1.,
            (self.likelihood_sum - self.likelihood_cache[last_i])
            / (self.likelihood_sum - self.likelihood_cache[next_i])
        )
        _log.debug(f"{A_mu_i_mu_star = }")

        # Acceptance / Rejection
        if u < A_mu_i_mu_star:
            _log.info(f"i = {self._i}: {last_i = } -> {next_i = }.")
            # Add to the history
            mu_star = self.models[next_i]
            self.history.append(mu_star)

            # Update the last_i
            self.last_i = next_i

        else:
            mu_i = self.history[-1]
            self.history.append(mu_i)

        return self.history[-1], last_i


class AlternativeGibbsPosteriorPiN(GibbsPosteriorPiN[TSample]):
    def __init__(
            self,
            data: list[TSample],
            models: list[DiscreteDistribution],
            seed=None,
            lazy_init=False,
    ):
        super().__init__(data, models, seed, True)

        self.models = [m for m in self.models if self.likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.random_state)
