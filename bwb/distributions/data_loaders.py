"""
This module contains the classes that are used to load the data.
"""
import abc
import time
import typing as t

import torch
import torchvision.transforms as T

import bwb.distributions as dist
import bwb.logging_ as logging
from bwb.config import config
from bwb.utils.protocols import ArrayLikeT

__all__ = [
    "BaseDistributionDataLoader",
    "DiscreteDistributionDataLoader",
    "DistributionDrawDataLoader",
]

_log = logging.get_logger(__name__)


class BaseDistributionDataLoader[DistributionT](
    t.MutableMapping[int, DistributionT], metaclass=abc.ABCMeta
):
    """
    Base class for DataLoaders. It is a :py:class:`MutableMapping` that
    creates instances of distributions in a 'lazy' way, saving
    computation time. It ends up representing several distributions
    from a tensor with the corresponding weights.
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
                "The sum over the dim 1 of the tensor "
                "probs_tensor must all be 1."
            )

        # Set the tensor of log-probabilities
        self.logits_tensor = torch.log(self.probs_tensor + config.eps)

        # And define the dictionary to wrap
        self._models: dict[int, DistributionT] = {i: None
                                                  for i in range(_n_probs)}

        toc = time.time()
        _log.debug(f"Δt={toc - tic:.2f} [seg]")

    @abc.abstractmethod
    def _create_distribution_instance(self, index) -> DistributionT:
        """To use template pattern on __get_item__"""
        raise NotImplementedError(
            "Must implement method '_create_distribution_instance'."
        )

    def __getitem__(self, item: int) -> DistributionT:
        if self._models[item] is None:
            self._models[item] = self._create_distribution_instance(item)
        return self._models[item]

    def __setitem__(self, key: int, value: DistributionT):
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
    BaseDistributionDataLoader[dist.DiscreteDistribution]):
    """
    DataLoader for the
    :py:class:`bwb.distributions.discrete_distributions.DiscreteDistributions`.
    """

    def _create_distribution_instance(
        self,
        index: int
    ) -> dist.DiscreteDistribution:
        return dist.DiscreteDistribution(self.probs_tensor[index])


class DistributionDrawDataLoader(
    BaseDistributionDataLoader[dist.DistributionDraw]
):
    """A class of type :py:class:`MutableMapping` that wraps a
    dictionary. It stores information from probability arrays and logits.
    This class can be thought of as using the flyweight
    pattern, so as not to take up too much instantiation time, or if an
    instance already exists, to reuse it."""

    def __init__(
        self,
        models_array: ArrayLikeT,
        original_shape,
        final_shape=None,
        floor=0,
    ):
        """
        :param models_array: Array of models. Each model is a tensor of
            probabilities.
        :param tuple[int, int] original_shape: Shape of the original
            images.
        :param int floor: The floor value for the probabilities.
        :param tuple[int, int] final_shape: The final shape of the
            images.
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
            T.Lambda(lambda x: x),  # Literally, do nothing
        ]
        if final_shape:
            list_transforms += [
                T.Lambda(lambda x: x.reshape(original_shape)),
                T.Lambda(lambda x: x / torch.max(x)),
                T.ToPILImage(),
                T.Resize(final_shape),
                T.ToTensor(),
                T.Lambda(lambda x: x / x.sum()),
                T.Lambda(lambda x: x.reshape(-1)),
            ]
        self.transform = T.Compose(list_transforms)

        toc = time.time()
        _log.debug(f"Δt={toc - tic:.2f} [seg]")

        super(DistributionDrawDataLoader, self).__init__(
            probs_tensor=probs_tensor)

    def _create_distribution_instance(self, index) -> dist.DistributionDraw:
        weights = self.transform(self.probs_tensor[index])
        return dist.DistributionDraw(weights, self.shape)

    def compute_likelihood(self, data: ArrayLikeT, **kwargs) -> torch.Tensor:
        """
        Compute the probabilities of the data given the models.

        :param data: The data to compute the probabilities.
        :return: A tuple with the probabilities and the logits.
        """
        # Data with shape (1, n_data)
        data = torch.as_tensor(data, device=config.device).reshape(1, -1)

        # logits array with shape (n_models, n_support)
        logits_models = self.logits_tensor

        # Take the evaluations of the logits, resulting in a tensor of
        # shape (n_models, n_data)
        evaluations = torch.take_along_dim(logits_models, data, 1)

        # Get the likelihood as cache. The shape is (n_models,)
        likelihood_cache = torch.exp(torch.sum(evaluations, 1))

        # Get the posterior probabilities.
        probabilities = likelihood_cache / likelihood_cache.sum()

        return probabilities

    def get(self, i: int, **kwargs) -> dist.DistributionDraw:
        """Get the model at the index ``i``."""
        return self[i]
