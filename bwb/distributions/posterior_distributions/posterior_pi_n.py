import abc
import functools
import time
from collections import Counter
from typing import Sequence

import numpy as np
import torch

from bwb.config import Config
from bwb.distributions.discrete_distribution import DiscreteDistribution
from bwb.validation import check_is_fitted

config = Config()


def _log_likelihood_default(model: DiscreteDistribution, data: torch.Tensor):
    """ Default log-likelihood of the posterior.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


def timeit_to_total_time(method):
    """Function that records the total time of a method and stores it in the class instance."""

    @functools.wraps(method)
    def timeit_wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = method(*args, **kwargs)
        toc = time.perf_counter()
        args[0].total_time += toc - tic
        return result

    return timeit_wrapper


# noinspection PyAttributeOutsideInit
class PosteriorPiN(abc.ABC):
    r"""
    Clase base para las clases que representen la distribución posterior:
     .. math::
        \Pi_n(dm) = \Pi(dm) \frac{\mathcal{L}_n(m)}{\int_{\mathcal{M}} \mathcal{L}_n(d\bar m) \Pi(d\bar m)}
    """

    def __init__(
            self,
            log_likelihood_fn,
            device,
    ):
        """
        Initiliser.

        :param log_likelihood_fn: The log-likelihood function. Is a function that recieves a model and data, and
        returns the log-likelihood.
        :param device: The device of the tensors.
        """
        # Set the device of the instance
        self.device: torch.device = torch.device(device or config.device)
        self.log_likelihood_fn = log_likelihood_fn or _log_likelihood_default

        # The history of the index samples
        self.samples_history: list[int] = []
        self.samples_counter: Counter = Counter()

        # To register the total time of the samples
        self.total_time = 0.

    def fit(
            self,
            data: Sequence[int],
            models: Sequence[DiscreteDistribution],
    ):
        """
        Fit the data of the distribution

        :param data: The data. It must be the indices of the discrete distribution.
        :param models: A sequence of models.
        :return: itself.
        """
        self.data_ = torch.as_tensor(data, device=self.device)
        self.models_ = np.asarray(models)
        self.models_index_ = np.arange(len(self.models_))
        return self

    def log_likelihood(self, model: DiscreteDistribution):
        """The log-likelehood of the model."""
        check_is_fitted(self)
        return self.log_likelihood_fn(model, data=self.data_)

    def likelihood(self, model: DiscreteDistribution):
        """The likelihood of the model."""
        return torch.exp(self.log_likelihood(model))

    @abc.abstractmethod
    def _draw(self, *args, **kwargs):
        """To use template pattern on ``draw`` method."""
        ...

    @abc.abstractmethod
    def _rvs(self, size=1, *args, **kwargs):
        """To use template pattern on ``_rvs`` method."""
        ...

    @timeit_to_total_time
    def draw(self, *args, **kwargs) -> DiscreteDistribution:
        """Draw a sample."""
        check_is_fitted(self)
        to_return, i = self._draw(*args, **kwargs)
        self.samples_history.append(i)
        self.samples_counter[i] += 1
        return to_return

    @timeit_to_total_time
    def rvs(self, size=1, *args, **kwargs) -> Sequence[DiscreteDistribution]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        check_is_fitted(self)
        to_return, list_i = self._rvs(size=size, *args, **kwargs)
        self.samples_history += list_i
        self.samples_counter.update(list_i)
        return to_return
