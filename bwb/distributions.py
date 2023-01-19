from __future__ import annotations

import abc
from itertools import product

import PIL.Image
import numpy as np
from PIL.Image import Image


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

    def obtain_rng(self, random_state):
        """Obtain a random state. If the random state is `None`, returns the generator of the class"""
        return self.rng if random_state is None else np.random.default_rng(random_state)


class DistributionDraw(RvsDistribution):

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

    def pdf(self, value: tuple[int, int] | list[tuple[int, int], ...]):
        """Probability density function."""
        if isinstance(value, tuple):
            value = [value]
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

        # Data = {x_i}_{i=1}^n
        self.data: list[tuple[int, int], ...] = data
        self.n_data: int = len(self.data)
        # Models space M = {m_i}_{i=1}^M
        self.models: list[DistributionDraw] = models
        self.n_models = len(self.models)
        # Likelihood cache
        self._likelihood_cache: np.ndarray = None

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
        # List of evaluations
        evaluations = []
        # Iterate over every value of the data
        for i in range(self.n_data):
            evaluation = m.matrix[self.data[i]]
            # If the evaluation is zero, return 0 directly
            if evaluation == 0:
                return 0.
            evaluations.append(evaluation)

        # Return the product of evaluations
        return np.prod(evaluations)

    @property
    def likelihood_cache(self) -> np.ndarray:
        """Likelihood cache to avoid to compute every time a same distribution, in order to be the same order of
        the list `models`."""
        if not self._likelihood_cache:
            self._likelihood_cache = np.array([self.likelihood(m) for m in self.models])
        return self._likelihood_cache

    def rvs(self, size=1, random_state=None) -> list[DistributionDraw]:
        # Set a random state
        rng = self.obtain_rng(random_state)
        return [self.draw(random_state=rng) for _ in range(size)]

    def draw(self, random_state=None) -> DistributionDraw:
        ...


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
        self.counter: dict[int, int] = {}
        if not lazy_init:
            self._first_step(self.rng)

    def update_counter(self, i):
        self.counter.setdefault(i, 0)
        self.counter[i] += 1

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

    def _first_step(self, rng):
        if not self.history:
            # To avoid a measure with zero likelihood
            probabilities = (self.likelihood_cache > 0).astype(float)
            probabilities /= probabilities.sum()
            # Choose a model
            self.last_i = int(rng.choice(self.n_models, p=probabilities))
            self.history.append(self.models[self.last_i])
            # Update the counter
            self.update_counter(self.last_i)

    def draw(self, random_state=None):
        # Set generator
        rng = self.obtain_rng(random_state)

        # If this is the first time that executes this instruction
        self._first_step(rng)
        last_i = self.last_i

        # Draw a uniform
        u = rng.uniform(low=0, high=1)

        # Draw a candidate
        possible_faces = list(range(self.n_models))
        possible_faces.remove(last_i)  # Remove the last face
        next_i = int(rng.choice(possible_faces))

        # Compute the acceptance probability
        A_mu_i_mu_star = min(
            1.,
            (self.likelihood_cache[next_i])
            / (self.likelihood_cache[last_i])
        )

        # Acceptance / Rejection
        if u < A_mu_i_mu_star:
            # Add to the history
            mu_star = self.models[next_i]
            self.history.append(mu_star)

            # Update the last_i
            self.last_i = next_i

        else:
            # Repeat the last sample
            mu_i = self.history[-1]
            self.history.append(mu_i)

        # Update the counter
        self.update_counter(self.last_i)

        return self.history[-1]


class GibbsPosteriorPiN(MCMCPosteriorPiN):

    def __init__(
            self,
            data: list[tuple[int, int], ...],
            models: list[DistributionDraw],
            seed=None,
            lazy_init=False,
    ):
        super().__init__(data, models, seed, lazy_init)
        self.possible_models = list(range(self.n_models))
        self.likelihood_sum: float = 0.
        self.probs_cache: dict[int, np.ndarray] = dict()

    def _first_step(self, rng):
        if not self.history:
            self.likelihood_sum = sum(self.likelihood_cache)
            # Choose a model
            self.last_i = int(rng.choice(self.n_models))
            self.history.append(self.models[self.last_i])
            # Update the counter
            self.update_counter(self.last_i)

    def draw(self, random_state=None):
        # Set generator
        rng = self.obtain_rng(random_state)

        # If this is the first time that executes this instruction
        self._first_step(rng)
        last_i = self.last_i

        # Draw a uniform
        u = rng.uniform(low=0, high=1)

        # Draw a candidate
        if last_i not in self.probs_cache:
            self.probs_cache[last_i] = np.array(
                self.likelihood_cache[i]
                / (self.likelihood_sum - self.likelihood_cache[last_i])
                for i in self.possible_models
            )
            self.probs_cache[last_i][last_i] = 0.

        next_i = int(rng.choice(self.possible_models, p=self.probs_cache[last_i]))

        # Compute the acceptance probability
        A_mu_i_mu_star = min(
            1.,
            (self.likelihood_sum - self.likelihood_cache[last_i])
            / (self.likelihood_sum - self.likelihood_cache[next_i])
        )

        # Acceptance / Rejection
        if u < A_mu_i_mu_star:
            # Add to the history
            mu_star = self.models[next_i]
            self.history.append(mu_star)

            # Update the last_i
            self.last_i = next_i

        else:
            mu_i = self.history[-1]
            self.history.append(mu_i)

        # Update the counter
        self.update_counter(last_i)

        return self.history[-1]
