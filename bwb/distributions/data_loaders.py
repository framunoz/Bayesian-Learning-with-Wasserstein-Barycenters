import abc
import time
import typing

import torch
import torchvision.transforms as transforms

import bwb.distributions as distrib
from bwb import logging
from bwb.config import config
from bwb.utils import _ArrayLike, _DistributionT

__all__ = [
    "BaseDistributionDataLoader",
    "DiscreteDistributionDataLoader",
    "DistributionDrawDataLoader",
]

_log = logging.get_logger(__name__)


class BaseDistributionDataLoader(
    typing.MutableMapping[int, _DistributionT], typing.Generic[_DistributionT], abc.ABC
):
    """
    Base class for DataLoaders. It is a :py:class:`MutableMapping` that creates instances of
    distributions in a 'lazy' way, saving computation time. It ends up representing several
    distributions from a tensor with the corresponding weights.
    """

    def __init__(
        self,
        probs_tensor,
    ):
        _log.debug("Creating a BaseDistributionDataLoader instance.")
        tic = time.time()
        # Set the probs_tensor
        self.probs_tensor: torch.Tensor = torch.as_tensor(
            probs_tensor, device=config.device, dtype=config.dtype
        )
        _n_probs = len(self.probs_tensor)
        probs_tensor_sum = self.probs_tensor.sum(dim=1)
        if not (
            torch.isclose(probs_tensor_sum, torch.ones_like(probs_tensor_sum))
        ).all():
            raise ValueError(
                "The sum over the dim 1 of the tensor probs_tensor must all be 1."
            )

        # Set the tensor of log-probabilities
        self.logits_tensor = torch.log(self.probs_tensor + config.eps)

        # And define the dictionary to wrap
        self._models: dict[int, _DistributionT] = {i: None for i in range(_n_probs)}

        toc = time.time()
        _log.debug(f"Δt={toc - tic:.2f} [seg]")

    @abc.abstractmethod
    def _create_distribution_instance(self, index) -> _DistributionT:
        """To use template pattern on __get_item__"""
        raise NotImplementedError(
            "Must implement method '_create_distribution_instance'."
        )

    def __getitem__(self, item: int) -> _DistributionT:
        if self._models[item] is None:
            self._models[item] = self._create_distribution_instance(item)
        return self._models[item]

    def __setitem__(self, key: int, value: _DistributionT):
        self._models[key] = value

    def __delitem__(self, key: int):
        del self._models[key]

    def __iter__(self):
        return self._models.__iter__()

    def __contains__(self, item):
        return item in self._models

    def __len__(self) -> int:
        return len(self._models)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + f"(n_models, n_supp)={tuple(self.probs_tensor.shape)}"
            + ")"
        )


class DiscreteDistributionDataLoader(
    BaseDistributionDataLoader[distrib.DiscreteDistribution]
):
    """
    DataLoader for the :py:class:`bwb.distributions.discrete_distributions.DiscreteDistributions`.
    """

    def _create_distribution_instance(self, index: int) -> distrib.DiscreteDistribution:
        return distrib.DiscreteDistribution(self.probs_tensor[index])


class DistributionDrawDataLoader(BaseDistributionDataLoader[distrib.DistributionDraw]):
    """A class of type :py:class:`MutableMapping` that wraps a dictionary. It stores information
    from probability arrays and logits. This class can be thought of as using the flyweight
    pattern, so as not to take up too much instantiation time, or if an instance already exists,
    to reuse it."""

    def __init__(
        self,
        models_array: _ArrayLike,
        original_shape,
        final_shape=None,
        floor=0,
    ):
        """
        :param models_array: Arreglo de modelos.
        :param tuple[int, int] shape: Dimensiones de las imágenes originales.
        :param int floor: Número que funciona como valor mínimo de las imágenes.
        """
        _log.debug("Creating a DistributionDrawDataLoader instance.")
        tic = time.time()
        self.floor: int = int(floor)

        # Setting the shape
        self.shape = original_shape if final_shape is None else final_shape

        # Setting the tensor of probabilities
        probs_tensor = torch.tensor(
            models_array, device=config.device, dtype=config.dtype
        )
        if self.floor != 0:
            probs_tensor = torch.min(probs_tensor, self.floor)
        probs_tensor = probs_tensor / 255
        probs_tensor = probs_tensor / torch.sum(probs_tensor, 1).reshape(-1, 1)

        # Set transforms
        list_transforms = [
            transforms.Lambda(lambda x: x),  # Literally, do nothing
        ]
        if final_shape:
            list_transforms += [
                transforms.Lambda(lambda x: x.reshape(original_shape)),
                transforms.Lambda(lambda x: x / torch.max(x)),
                transforms.ToPILImage(),
                transforms.Resize(final_shape),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x / x.sum()),
                transforms.Lambda(lambda x: x.reshape(-1)),
            ]
        self.transform = transforms.Compose(list_transforms)

        toc = time.time()
        _log.debug(f"Δt={toc - tic:.2f} [seg]")

        super(DistributionDrawDataLoader, self).__init__(probs_tensor=probs_tensor)

    def _create_distribution_instance(self, index) -> distrib.DistributionDraw:
        weights = self.transform(self.probs_tensor[index])
        return distrib.DistributionDraw.from_weights(weights, self.shape)
