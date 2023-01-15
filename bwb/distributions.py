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
        """Matrix representation of the image."""
        if self._matrix is None:
            image_as_array = np.asarray(self.image)
            if len(image_as_array.shape) == 2:
                matrix = image_as_array
            else:
                matrix = image_as_array[:, :, 0]
            matrix = 1 - (matrix / 255)
            self._matrix = matrix / matrix.sum()
        return self._matrix

    def rvs(self, size=1, random_state=None) -> list[tuple[int, int]]:
        # Set a random state
        rng = self._rng if random_state is None else np.random.default_rng(random_state)
        n, m = self.matrix.shape
        indices: list[tuple[int, int], ...] = [(i, j) for i, j in product(range(n), range(m))]
        # Sample with respect the weight matrix.
        samples = rng.choice(a=len(indices), size=size, p=self.matrix.flatten())
        return [indices[s] for s in samples]

    def pdf(self, *args, **kwargs):
        return self.matrix[args]
