from __future__ import annotations

import abc
from itertools import product
from typing import List, Tuple

import PIL.Image
import numpy as np
from PIL.Image import Image


class RvsDistribution(abc.ABC):
    """
    This class provides the interface for those distributions that can be sampled from them.
    """

    @abc.abstractmethod
    def rvs(self, *args, **kwargs):
        """Random variates."""
        ...


class Distribution(RvsDistribution, abc.ABC):
    """
    This class provides try to copy the interface of an instance of a distribution class of scipy.
    """

    @abc.abstractmethod
    def pdf(self, *args, **kwargs):
        """Probability density function."""
        ...


class DistributionDraw(Distribution):

    def __init__(self, image: Image, seed=None):
        """
        Initializer

        :param image: The image to obtain the distribution.
        """
        self._image: Image = image
        self._matrix: np.ndarray = None
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._support: list[tuple[int, int], ...] = None

        # Indices
        self._indices: list[tuple[int, int], ...] = None
        self._indices_inv: dict[tuple[int, int], int] = None

    @classmethod
    def fromarray(cls, array: np.ndarray, seed=None):
        """
        Alternative initializer.

        :param array: An array of integers with minimum value 0 and maximum value 255.
        :param seed:
        :return:
        """
        image = PIL.Image.fromarray(array)
        return cls(image=image, seed=seed)

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
    def shape(self):
        """The shape of the matrix"""
        return self.matrix.shape

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
        rng = self._rng if random_state is None else np.random.default_rng(random_state)
        # Sample with respect the weight matrix.
        samples = rng.choice(a=len(self.indices), size=size, p=self.matrix.flatten())
        return [self.indices[s] for s in samples]

    def pdf(self, value: tuple[int, int] | list[tuple[int, int], ...]):
        if isinstance(value, tuple):
            value = [value]
        indices_from_coord = [self.indices_inv[tup] for tup in value]
        return self.matrix.take(indices_from_coord)

    def _repr_png_(self):
        """iPython display hook support

        :returns: png version of the image as bytes
        """
        return self.image._repr_png_()


class DistributionDrawBuilder:
    """
    Builder for the class DistributionDraw.
    """

    def __init__(self, seed=None):
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def create(self, image: Image) -> DistributionDraw:
        return DistributionDraw(image=image, seed=self._rng)

    def create_fromarray(self, array: np.ndarray) -> DistributionDraw:
        return DistributionDraw.fromarray(array=array, seed=self._rng)

    def set_rng(self, rng: np.random.Generator):
        self._rng: np.random.Generator = np.random.default_rng(rng)