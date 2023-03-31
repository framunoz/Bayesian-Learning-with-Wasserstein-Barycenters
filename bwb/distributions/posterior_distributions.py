import abc
import functools
import time
from collections import Counter
from typing import Sequence

import numpy as np
import torch
from numpy.random import Generator

from bwb.config import Config
from bwb.distributions.discrete_distribution import DiscreteDistribution, BaseDistributionDataLoader
from bwb.validation import check_is_fitted

__all__ = [
    "PosteriorPiN",
    "ExplicitPosteriorPiN",
]


def _log_likelihood_default(model: DiscreteDistribution, data: torch.Tensor):
    """ Default log-likelihood of the posterior.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


def timeit_to_total_time(method):
    """Function that records the total time it takes to execute a method, and stores it in the ``total_time``
    attribute of the class instance."""

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
    r"""Base class for classes representing the posterior distribution:

    .. math::
        \Pi_n(dm) = \Pi(dm) \frac{\mathcal{L}_n(m)}{\int_{\mathcal{M}} \mathcal{L}_n(d\bar m) \Pi(d\bar m)}

    """

    def __init__(
            self,
            log_likelihood_fn,
    ):
        """
        Initiliser.

        :param log_likelihood_fn: The log-likelihood function. Is a function that receives a model and data, and
        returns the log-likelihood.
        """
        self.log_likelihood_fn = log_likelihood_fn or _log_likelihood_default

        # The history of the index samples
        self.samples_history: list[int] = []
        self.samples_counter: Counter = Counter()

        # To register the total time of the samples
        self.total_time = 0.

    def fit(
            self,
            data: Sequence[int],
            models: BaseDistributionDataLoader,
    ):
        """
        Fit the data of the distribution

        :param data: The data. It must be the indices of the discrete distribution.
        :param models: A sequence of models.
        :return: itself.
        """
        self.data_ = torch.as_tensor(data, device=Config.device)
        self.models_: BaseDistributionDataLoader = models
        self.models_index_ = np.arange(len(self.models_))
        return self

    def log_likelihood(self, model: DiscreteDistribution):
        """The log-likelihood of the model."""
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


# noinspection PyAttributeOutsideInit
class ExplicitPosteriorPiN(PosteriorPiN):
    r"""Distribution that uses the strategy of calculating all likelihoods by brute force. This class implements
    likelihoods of the form

    .. math::
        \mathcal{L}_n(m) = \prod_{i=1}^{n} \rho_{m}(x_i)

    using the log-likelihood for stability. Finally, to compute the sampling probabilities, for a discrete
    set :math:`\mathcal{M}` of models, using a uniform prior, we have the posterior explicit by

    .. math::
        \Pi_n(m) = \frac{\mathcal{L}_n(m)}{\sum_{\bar m \in \mathcal{M}} \mathcal{L}_n(\bar m)}

     """

    def __init__(
            self,
            log_likelihood_fn=None,
    ):
        super().__init__(log_likelihood_fn)

    def fit(
            self,
            data: Sequence[int],
            models: BaseDistributionDataLoader,
    ):
        super(ExplicitPosteriorPiN, self).fit(data, models)

        # Compute the log-likelihood of the models as cache
        # Data with shape (1, n_data)
        data = self.data_.reshape(1, -1)

        # logits array with shape (n_models, n_support)
        logits_models = models.logits_tensor

        # Take the evaluations of the logits, resulting in a tensor of shape (n_models, n_data)
        evaluations = torch.take_along_dim(logits_models, data, 1)

        # Get the likelihood as cache. The shape is (n_models,)
        likelihood_cache = torch.exp(torch.sum(evaluations, 1))

        # Get the posterior probabilities.
        self.probabilities_: np.ndarray = (likelihood_cache / likelihood_cache.sum()).cpu().numpy()

        return self

    def _draw(self, seed=None):
        rng: Generator = np.random.default_rng(seed)
        i = rng.choice(a=self.models_index_, p=self.probabilities_)
        return self.models_[i], i

    def _rvs(self, size=1, seed=None, **kwargs):
        rng: Generator = np.random.default_rng(seed)
        list_i = list(rng.choice(a=self.models_index_, size=size, p=self.probabilities_))
        return [self.models_[i] for i in list_i], list_i

    def draw(self, seed=None, **kwargs):
        super().draw(seed=seed)

    def rvs(self, size=1, seed=None, **kwargs):
        super().rvs(size=size, seed=seed)
