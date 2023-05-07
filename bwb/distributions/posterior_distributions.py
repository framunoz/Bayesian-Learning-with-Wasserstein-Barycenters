import abc
import collections
import functools
import time
import typing

import numpy as np
import torch

import bwb.distributions as distrib
import bwb.distributions.data_loaders as data_loaders
import bwb.validation as validation
from bwb.config import config
from bwb.utils import _ArrayLike, _DistributionT

__all__ = [
    "PosteriorPiN",
    "ExplicitPosteriorPiN",
]


def _log_likelihood_default(model: distrib.DiscreteDistribution, data: torch.Tensor):
    """ Default log-likelihood of the posterior.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


def _timeit_to_total_time(method):
    """Function that records the total time it takes to execute a method, and stores it in the
    ``total_time`` attribute of the class instance."""

    @functools.wraps(method)
    def timeit_wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = method(*args, **kwargs)
        toc = time.perf_counter()
        args[0].total_time += toc - tic
        return result

    return timeit_wrapper


# noinspection PyAttributeOutsideInit
class PosteriorPiN(abc.ABC, typing.Generic[_DistributionT]):
    r"""Base class for classes representing the posterior distribution:

    .. math::
        \Pi_n(dm)
        = \Pi(dm) \frac{\mathcal{L}_n(m)}{\int_{\mathcal{M}} \mathcal{L}_n(d\bar m) \Pi(d\bar m)}

    """

    def __init__(
            self,
            log_likelihood_fn,
    ):
        """
        Initialiser.

        :param log_likelihood_fn: The log-likelihood function. Is a function that receives a model
        and data, and returns the log-likelihood.
        """
        self.log_likelihood_fn = log_likelihood_fn or _log_likelihood_default

        # The history of the index samples
        self.samples_history: list[int] = []
        self.samples_counter: collections.Counter = collections.Counter()

        # To register the total time of the samples
        self.total_time = 0.

        self.__fitted = False

    def fit(
            self,
            data: _ArrayLike,
            models: data_loaders.BaseDistributionDataLoader[_DistributionT],
    ):
        """
        Fit the data of the distribution

        :param data: The data. It must be the indices of the discrete distribution.
        :param models: A sequence of models.
        :return: itself.
        """
        self.data_ = torch.as_tensor(data, device=config.device)
        self.models_: data_loaders.BaseDistributionDataLoader[_DistributionT] = models
        self.models_index_ = np.arange(len(self.models_))
        self.__fitted = True
        return self

    def log_likelihood(self, model: _DistributionT):
        """The log-likelihood of the model."""
        validation.check_is_fitted(self)
        return self.log_likelihood_fn(model, data=self.data_)

    def likelihood(self, model: _DistributionT):
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

    @_timeit_to_total_time
    def draw(self, *args, **kwargs) -> _DistributionT:
        """Draw a sample."""
        validation.check_is_fitted(self)
        to_return, i = self._draw(*args, **kwargs)
        self.samples_history.append(i)
        self.samples_counter[i] += 1
        return to_return

    @_timeit_to_total_time
    def rvs(self, size=1, *args, **kwargs) -> typing.Sequence[_DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        validation.check_is_fitted(self)
        to_return, list_i = self._rvs(size=size, *args, **kwargs)
        self.samples_history += list_i
        self.samples_counter.update(list_i)
        return to_return

    def __repr__(self):
        if self.__fitted:
            return (self.__class__.__name__
                    + "("
                    + f"n_data={len(self.data_)}, "
                    + f"n_models={len(self.models_)}"
                    + ")")
        return self.__class__.__name__ + "()"


# noinspection PyAttributeOutsideInit
class ExplicitPosteriorPiN(PosteriorPiN[_DistributionT]):
    r"""Distribution that uses the strategy of calculating all likelihoods by brute force. This
    class implements likelihoods of the form

    .. math::
        \mathcal{L}_n(m) = \prod_{i=1}^{n} \rho_{m}(x_i)

    using the log-likelihood for stability. Finally, to compute the sampling probabilities, for a
    discrete set :math:`\mathcal{M}` of models, using a uniform prior, we have the posterior
    explicit by

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
            data,
            models,
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

    def _draw(self, seed=None) -> _DistributionT:
        rng: np.random.Generator = np.random.default_rng(seed)
        i = rng.choice(a=self.models_index_, p=self.probabilities_)
        return self.models_[i], i

    def _rvs(self, size=1, seed=None, **kwargs) -> typing.Sequence[_DistributionT]:
        rng: np.random.Generator = np.random.default_rng(seed)
        list_i = list(rng.choice(a=self.models_index_, size=size, p=self.probabilities_))
        return [self.models_[i] for i in list_i], list_i
