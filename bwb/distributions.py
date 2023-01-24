from __future__ import annotations

import abc
import logging
from collections import Counter
from itertools import product

import PIL.Image
import numpy as np
from PIL.Image import Image
from numpy.random import Generator


# Get logger
_log = logging.getLogger(__name__)
_log.setLevel(logging.WARNING)

# Configure a formatter
_formatter = logging.Formatter("%(asctime)s: %(levelname)s [%(name)s:%(lineno)s] %(message)s")

# Configure a file handler
_file_handler = logging.FileHandler(f"logging.log")
_file_handler.setFormatter(_formatter)
_log.addHandler(_file_handler)

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
_log.addHandler(_stream_handler)


class RvsDistribution(abc.ABC):
    """
    This class provides the interface for those distributions that can be sampled from them.
    """

    def __init__(self, seed):
        # Generator
        self.rng: np.random.Generator = np.random.default_rng(seed)

    @abc.abstractmethod
    def rvs(self, *args, **kwargs):
        """Random variates."""
        ...

    @abc.abstractmethod
    def draw(self, *args, **kwargs):
        """To draw a sample from the distribution."""
        ...

    def obtain_rng(self, random_state) -> np.random.Generator:
        """Obtain a random state. If the random state is `None`, returns the generator of the class"""
        return self.rng if random_state is None else np.random.default_rng(random_state)


class Distribution(RvsDistribution, abc.ABC):
    @abc.abstractmethod
    def pdf(self, *args, **kwargs) -> float | list[float]:
        """Probability density function."""
        ...


class DistributionDraw(Distribution):

    def __init__(self, image: Image, seed=None):
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
        self._support: list[tuple[int, int], ...] = None

        # Indices
        self._indices: list[tuple[int, int], ...] = None
        self._indices_inv: dict[tuple[int, int], int] = None

    @classmethod
    def fromarray(cls, array: np.ndarray, seed=None, ceil=0):
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
    def shape(self):
        """The shape of the matrix"""
        return self.matrix.shape

    @property
    def indices(self) -> list[tuple[int, int], ...]:
        """A list of coordinates of the domain."""
        if self._indices is None:
            n, m = self.shape
            self._indices = [(i, j) for i, j in product(range(n), range(m))]
        return self._indices

    @property
    def indices_inv(self) -> dict[tuple[int, int], int]:
        """The inverse function that return the index if the component."""
        if self._indices_inv is None:
            self._indices_inv: dict[tuple[int, int], int] = {val: k for k, val in enumerate(self.indices)}
        return self._indices_inv

    @property
    def support(self) -> list[tuple[int, int], ...]:
        """Support of the distribution."""
        if self._support is None:
            non_zero_coord: np.ndarray = np.array(np.nonzero(self.matrix)).T
            self._support: list[tuple[int, int], ...] = [tuple(row) for row in non_zero_coord]
        return self._support

    @property
    def weights(self):
        """Return its correspondient weights in the same order of it support."""
        return self.pdf(self.support)

    def rvs(self, size=1, random_state=None) -> list[tuple[int, int]]:
        # Set a random state
        rng = self.obtain_rng(random_state)
        # Sample with respect the weight matrix.
        samples = rng.choice(a=len(self.indices), size=size, p=self.matrix.flatten())
        return [self.indices[s] for s in samples]

    def draw(self, random_state=None) -> tuple[int, int]:
        return self.rvs(size=1, random_state=random_state)[0]

    def pdf(self, value: tuple[int, int] | list[tuple[int, int], ...]) -> float | list[float]:
        if isinstance(value, tuple):
            return self.matrix[value]
        indices_from_coord = [self.indices_inv[tup] for tup in value]
        return self.matrix.take(indices_from_coord)


class DistributionDrawBuilder:
    """
    Builder for the class DistributionDraw.
    """

    def __init__(self, seed=None, ceil=0):
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._ceil = ceil

        self._dict_kwargs = dict(seed=self._rng, ceil=self._ceil)

    def create(self, image: Image) -> DistributionDraw:
        return DistributionDraw(image=image, **self._dict_kwargs)

    def create_fromarray(self, array: np.ndarray) -> DistributionDraw:
        return DistributionDraw.fromarray(array=array, **self._dict_kwargs)

    def set_rng(self, rng: np.random.Generator):
        self._rng: np.random.Generator = np.random.default_rng(rng)
        self._dict_kwargs["seed"] = self._rng
        return self

    def set_ceil(self, ceil: int):
        self._ceil = ceil
        self._dict_kwargs["ceil"] = self._ceil
        return self


class PosteriorPiN(RvsDistribution, abc.ABC):
    """Abstract class used as protocol for the distributions used as Posterior."""

    def __init__(
            self,
            data: list[tuple[int, int], ...],
            models: list[DistributionDraw],
            seed,
    ):
        super().__init__(seed)

        # Number of the iteration
        self._i = 0

        # Data = {x_i}_{i=1}^n
        self.data: list[tuple[int, int], ...] = data
        # Models space M = {m_i}_{i=1}^M
        self.models: list[DistributionDraw] = models
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
    def likelihood_cache(self) -> np.ndarray:
        """Likelihood cache to avoid to compute every time a same distribution, in order to be the same order of
        the list `models`."""
        if self._likelihood_cache is None or len(self._likelihood_cache) != self.n_models:
            _log.debug(f"Calculating the likelihood cache...")
            self._likelihood_cache = np.array([self.likelihood(m) for m in self.models])
            _log.debug(f"Likelihood cache calculated.")
        return self._likelihood_cache

    def __repr__(self):
        return self.__class__.__name__ + f"(n_data={self.n_data}, n_models={self.n_models})"

    def likelihood(self, m: DistributionDraw) -> float:
        """
        Likelihood of the function:

        .. math:: \mathcal{L}_n(m) = \prod_{j=1}^{n} f_m(x_j)

        Parameters
        ----------
        m: DistributionDraw
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

    def rvs(self, size=1, random_state=None) -> list[DistributionDraw]:
        # Set a random state
        rng = self.obtain_rng(random_state)
        return [self.draw(random_state=rng) for _ in range(size)]

    def draw(self, random_state=None) -> DistributionDraw:
        _log.debug("=" * 10 + f" i = {self._i} " + "=" * 10)
        self._i += 1
        draw, i = self._draw(random_state=random_state)
        self.counter[i] += 1
        return draw

    @abc.abstractmethod
    def _draw(self, random_state) -> tuple[DistributionDraw, int]:
        """To use template pattern. Returns a distribution and its index."""
        ...

    def most_common(self, n=None) -> list[DistributionDraw]:
        return [self.models[k] for k, _ in self.counter.most_common(n)]


class MCMCPosteriorPiN(PosteriorPiN, abc.ABC):
    """Abstract class for the implementations of the posterior, using MCMC algorithms."""

    def __init__(
            self,
            data: list[tuple[int, int], ...],
            models: list[DistributionDraw],
            seed,
            lazy_init,
    ):
        super().__init__(data, models, seed)
        # History {mu^(i)}_{i=1}^N
        self.history: list[DistributionDraw] = []
        self.last_i: int = None
        if not lazy_init:
            self._first_step(self.rng)

    def __repr__(self):
        return (self.__class__.__name__ + f"(n_data={self.n_data}, n_models={self.n_models}, "
                f"n_samples={len(self.history) - 1}, last_i={self.last_i})")

    @abc.abstractmethod
    def _first_step(self, rng):
        ...


class MetropolisPosteriorPiN(MCMCPosteriorPiN):

    def __init__(
            self,
            data: list[tuple[int, int], ...], models: list[DistributionDraw],
            seed=None, lazy_init=False
    ):
        super().__init__(data, models, seed, lazy_init)
        self.possible_models = np.arange(self.n_models)

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

    def _draw(self, random_state=None) -> tuple[DistributionDraw, int]:
        # Set generator
        rng = self.obtain_rng(random_state)

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
            _log.debug(f"{last_i = } -> {next_i = }.")
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


class AlternativeMetropolisPosteriorPiN(MetropolisPosteriorPiN):
    def __init__(
            self,
            data: list[tuple[int, int], ...], models: list[DistributionDraw],
            seed=None, lazy_init=False
    ):
        super().__init__(data, models, seed, True)
        self.models = [m for m in self.models if self.likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.rng)


class GibbsPosteriorPiN(MCMCPosteriorPiN):

    def __init__(
            self,
            data: list[tuple[int, int], ...],
            models: list[DistributionDraw],
            seed=None,
            lazy_init=False,
    ):
        super().__init__(data, models, seed, True)
        self.possible_models = np.arange(self.n_models)
        self.probs_cache: dict[int, np.ndarray] = dict()
        self.likelihood_sum: float = 0.

        if not lazy_init:
            self._first_step(self.rng)

    def __repr__(self):
        return (self.__class__.__name__ + f"(n_data={self.n_data}, n_models={self.n_models}, "
                f"n_samples={len(self.history) - 1}, last_i={self.last_i}, likelihood_sum={self.likelihood_sum})")

    def _first_step(self, rng):
        _log.info("Executing _first_step")
        self.likelihood_sum = np.sum(self.likelihood_cache)
        _log.debug(f"likelihood_sum = {self.likelihood_sum}")
        # Choose a model
        possible_distributions = np.arange(self.n_models)[self.likelihood_cache > 0]
        self.last_i = int(rng.choice(possible_distributions))
        _log.info(f"First model selected: {self.last_i}.")
        self.history.append(self.models[self.last_i])
        # Update the counter
        self.counter[self.last_i] += 1

    def _draw(self, random_state=None) -> tuple[DistributionDraw, int]:
        # Set generator
        rng = self.obtain_rng(random_state)

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
        _log.debug(f"{self.likelihood_cache[next_i] = }")

        # Acceptance / Rejection
        if u < A_mu_i_mu_star:
            _log.debug(f"{last_i = } -> {next_i = }.")
            # Add to the history
            mu_star = self.models[next_i]
            self.history.append(mu_star)

            # Update the last_i
            self.last_i = next_i

        else:
            mu_i = self.history[-1]
            self.history.append(mu_i)

        return self.history[-1], last_i


class AlternativeGibbsPosteriorPiN(GibbsPosteriorPiN):
    def __init__(
            self,
            data: list[tuple[int, int], ...],
            models: list[DistributionDraw],
            seed=None,
            lazy_init=False,
    ):
        super().__init__(data, models, seed, True)

        self.models = [m for m in self.models if self.likelihood(m) > 0]
        self.possible_models = np.arange(self.n_models)

        if not lazy_init:
            self._first_step(self.rng)

