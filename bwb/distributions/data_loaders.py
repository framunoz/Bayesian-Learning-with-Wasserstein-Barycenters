import abc
from typing import TypeVar, MutableMapping, Generic

import torch

from bwb.config import config
from bwb.distributions import DiscreteDistribution, DistributionDraw

__all__ = [
    "BaseDistributionDataLoader",
    "DiscreteDistributionDataLoader",
    "DistributionDrawDataLoader",
]

TDistribution = TypeVar("TDistribution")


class BaseDistributionDataLoader(MutableMapping, Generic[TDistribution], abc.ABC):
    """
    Base class for DataLoaders. It is a ``MutableMapping`` that creates instances of distributions in a 'lazy' way,
    saving computation time. It ends up representing several distributions from a tensor with the corresponding weights.
    """

    def __init__(
            self,
            probs_tensor,
    ):
        # Set the probs_tensor
        self.probs_tensor: torch.Tensor = torch.as_tensor(probs_tensor, device=config.device)
        _n_probs = len(self.probs_tensor)
        probs_tensor_sum = self.probs_tensor.sum(dim=1)
        if not (torch.isclose(probs_tensor_sum, torch.ones_like(probs_tensor_sum))).all():
            raise ValueError("The sum over the dim 1 of the tensor probs_tensor must all be 1.")

        # Set the tensor of log-probabilities
        self.logits_tensor = torch.logit(self.probs_tensor, config.eps)

        # And define the dictionary to wrap
        self._models: dict[int, TDistribution] = {i: None for i in range(_n_probs)}

    @abc.abstractmethod
    def _create_distribution_instance(self, index) -> TDistribution:
        """To use template pattern on __get_item__"""
        raise NotImplementedError("Must implement method '_create_distribution_instance'.")

    def __getitem__(self, item: int) -> TDistribution:
        if self._models[item] is None:
            self._models[item] = self._create_distribution_instance(item)
        return self._models[item]

    def __setitem__(self, key: int, value: TDistribution):
        self._models[key] = value

    def __delitem__(self, key: int):
        del self._models[key]

    def __iter__(self):
        return self._models.__iter__()

    def __contains__(self, item):
        return item in self._models

    def __len__(self) -> int:
        return len(self._models)


class DiscreteDistributionDataLoader(BaseDistributionDataLoader[DiscreteDistribution]):
    """
    DataLoader for the ``DiscreteDistributions``.
    """

    def _create_distribution_instance(self, index: int) -> DiscreteDistribution:
        return DiscreteDistribution(self.probs_tensor[index])


class DistributionDrawDataLoader(BaseDistributionDataLoader[DistributionDraw]):
    """A class of type ``MutableMapping`` that wraps a dictionary. It stores information from probability arrays and
     logits. This class can be thought of as using the flyweight pattern, so as not to take up too much instantiation
     time, or if an instance already exists, to reuse it."""

    def __init__(
            self,
            models_array,
            shape,
            floor=0,
    ):
        """
        :type models_array: scalar or sequence or arraylike

        :param models_array: Arreglo de modelos.
        :type shape: tuple[int, int]
        :param shape: Dimensiones de las imágenes originales.
        :type floor: int
        :param floor: Número que funciona como valor mínimo de las imágenes.
        """
        self.floor: int = int(floor)

        # Setting the shape
        self.shape = shape

        # Setting the tensor of probabilities
        probs_tensor = torch.tensor(models_array, device=config.device)
        if self.floor != 0:
            probs_tensor = torch.min(probs_tensor, self.floor)
        probs_tensor = probs_tensor / 255
        probs_tensor = probs_tensor / torch.sum(probs_tensor, 1).reshape(-1, 1)

        super(DistributionDrawDataLoader, self).__init__(probs_tensor=probs_tensor)

    def _create_distribution_instance(self, index) -> DistributionDraw:
        return DistributionDraw(self.probs_tensor[index], self.shape)
