"""
This module contains the base classes and protocols for the models.
"""
import abc
import typing as t

import torch
from torchvision.datasets import VisionDataset

from bwb.distributions.discrete_distribution import DistributionDraw
from bwb.utils import _ArrayLike

__all__ = [
    "PDiscreteModelsSet",
    "BaseDiscreteModelsSet",
    "PDiscreteWeightedModelSet",
    "BaseDiscreteWeightedModelSet",
    "ModelDataset",
]


@t.runtime_checkable
class PDiscreteModelsSet[DistributionT](t.Protocol):
    """
    Protocol for classes that are a set of models with a discrete support.
    """

    def get(self, i: int, **kwargs) -> DistributionT:
        """Get the model at the index ``i``."""
        ...

    def __len__(self) -> int:
        """Get the number of models."""
        ...


class BaseDiscreteModelsSet[DistributionT](metaclass=abc.ABCMeta):
    """
    Base class for a set of models with a discrete support.
    """

    @abc.abstractmethod
    def get(self, i: int, **kwargs) -> DistributionT:
        """Get the model at the index ``i``."""
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Get the number of models."""
        pass


@t.runtime_checkable
class PDiscreteWeightedModelSet[DistributionT](PDiscreteModelsSet[DistributionT], t.Protocol):
    """
    Protocol for classes that are a weighted set of models with a discrete support.
    """

    def compute_likelihood(self, data: _ArrayLike = None, **kwargs) -> torch.Tensor:
        """
        Compute the probabilities of the data given the models.

        :param data: The data to compute the probabilities.
        :return: A tensor with the probabilities.
        """
        ...


class BaseDiscreteWeightedModelSet[DistributionT](BaseDiscreteModelsSet[DistributionT], metaclass=abc.ABCMeta):
    """
    Base class for a weighted set of models with a discrete support.
    """

    @abc.abstractmethod
    def compute_likelihood(self, data: _ArrayLike = None, **kwargs) -> torch.Tensor:
        """
        Compute the probabilities of the data given the models.

        :param data: The data to compute the probabilities.
        :return: A tensor with the probabilities.
        """
        pass


class ModelDataset(BaseDiscreteModelsSet[DistributionDraw]):
    """
    An adapter class that adapts a torchvision.vision.VisionDataset to a BaseDiscreteModelsSet.
    """

    def __init__(self, dataset: VisionDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def get(self, i: int, **kwargs) -> DistributionDraw:
        return DistributionDraw.from_grayscale_weights(self.dataset[i][0])
