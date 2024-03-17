import abc
import collections as c
import functools
import time
import typing as t
import warnings

import torch

import bwb.distributions as dist
import bwb.validation as validation
from bwb.config import config
from bwb.utils import _ArrayLike, _DistributionT

__all__ = [
    "DistributionSampler",
    "DiscreteDistributionSampler",
    "ExplicitPosteriorSampler",
]


def _log_likelihood_default(model: dist.DiscreteDistribution, data: torch.Tensor):
    """Default log-likelihood of the posterior.

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


def _set_generator(seed=None, device="cpu") -> torch.Generator:
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
        return gen
    gen.manual_seed(seed)
    return gen


@t.runtime_checkable
class DiscreteModelsSet(t.Protocol, t.Generic[_DistributionT]):
    """
    Protocol for classes that are a set of models with a discrete support.
    """

    def _compute_probability(self, data: _ArrayLike, **kwargs) -> torch.Tensor:
        """
        Compute the probabilities of the data given the models.

        :param data: The data to compute the probabilities.
        :return: A tensor with the probabilities.
        """
        ...

    def _get(self, i: int, **kwargs) -> _DistributionT:
        """Get the model at the index ``i``."""
        ...

    def __len__(self) -> int:
        """Get the number of models."""
        ...


class DistributionSampler(abc.ABC, t.Generic[_DistributionT]):
    r"""
    Base class for distributions that sampling other distributions. i.e. it represents a distribution :math:`\Lambda(dm) \in \mathcal{P}(\mathcal{M)}`, where :math:`\mathcal{M}` is the set of models.
    """

    @abc.abstractmethod
    def draw(self, *args, **kwargs) -> _DistributionT:
        """Draw a sample."""
        ...

    @abc.abstractmethod
    def rvs(self, size=1, *args, **kwargs) -> t.Sequence[_DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class DiscreteDistributionSampler(DistributionSampler[_DistributionT]):
    r"""
    Base class for distributions that have a discrete set of models. i.e. where the set of models is :math:`|\mathcal{M}| < +\infty`.

    As the support is discrete, the distribution can be represented as a vector of probabilities, and therefore, the sampling process is reduced to drawing an index from a multinomial distribution. This property allows to save the samples and the number of times each model has been sampled, to get statistics about the sampling process.
    """

    def __init__(self, save_samples: bool = True):
        self.save_samples = save_samples
        self.samples_history: list[int] = []
        self.samples_counter: c.Counter[int] = c.Counter()
        self._models_cache: dict[int, _DistributionT] = {}
        self.total_time = 0.0

    @abc.abstractmethod
    def _draw(self, *args, **kwargs) -> tuple[_DistributionT, int]:
        """To use template pattern on the draw method."""
        ...

    @_timeit_to_total_time
    def draw(self, *args, **kwargs) -> _DistributionT:
        """Draw a sample."""
        validation.check_is_fitted(self)
        to_return, i = self._draw(*args, **kwargs)
        if self.save_samples:  # Register the sample
            self.samples_history.append(i)
            self.samples_counter[i] += 1
        return to_return

    @abc.abstractmethod
    def _rvs(
        self, size=1, *args, **kwargs
    ) -> tuple[t.Sequence[_DistributionT], list[int]]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        ...

    @_timeit_to_total_time
    def rvs(self, size=1, *args, **kwargs) -> t.Sequence[_DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        validation.check_is_fitted(self)
        to_return, list_indices = self._rvs(size, *args, **kwargs)
        if self.save_samples:  # Register the samples
            self.samples_history.extend(list_indices)
            self.samples_counter.update(list_indices)
        return to_return

    @abc.abstractmethod
    def get_model(self, i: int) -> _DistributionT:
        """Get the model with index i."""
        ...

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if self.save_samples:
            to_return += f"(samples={len(self.samples_history)})"

        return to_return


class ExplicitPosteriorSampler(DiscreteDistributionSampler[_DistributionT]):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fitted = False

    @_timeit_to_total_time
    def fit(
        self,
        data: _ArrayLike,
        models: DiscreteModelsSet[_DistributionT],
        batch_size: int = 256,
    ):
        """
        Fit the posterior distribution.

        :param data: The data to fit the posterior.
        :param models: The models to fit the posterior.
        :param batch_size: The batch size to compute the probabilities.
        :return: The fitted posterior.
        """
        assert isinstance(
            models, DiscreteModelsSet
        ), "The models must be a DiscreteModelsSet."

        self.data_: torch.Tensor = torch.as_tensor(data, device=config.device)
        self.models_: DiscreteModelsSet[_DistributionT] = models
        self.models_index_: torch.Tensor = torch.arange(
            len(models), device=config.device
        )

        data = self.data_.reshape(1, -1)

        self.probabilities_: torch.Tensor = models._compute_probability(
            data, batch_size=batch_size
        )

        self.support_ = self.models_index_[self.probabilities_ > 0]

        self._fitted = True

        return self

    def get_model(self, i: int) -> _DistributionT:
        """Get the model with index i."""
        validation.check_is_fitted(self)
        if self._models_cache.get(i) is None:
            self._models_cache[i] = self.models_._get(i)
        return self._models_cache[i]

    def _draw(self, seed=None, *args, **kwargs) -> tuple[_DistributionT, int]:
        rng: torch.Generator = _set_generator(seed=seed, device=config.device)

        i = torch.multinomial(
            input=self.probabilities_, num_samples=1, generator=rng
        ).item()
        i = int(i)
        return self.get_model(i), i

    def _rvs(
        self, size=1, seed=None, *args, **kwargs
    ) -> tuple[t.Sequence[_DistributionT], list[int]]:
        rng: torch.Generator = _set_generator(seed=seed, device=config.device)

        indices = torch.multinomial(
            input=self.probabilities_, num_samples=size, replacement=True, generator=rng
        )
        indices = indices.tolist()
        return [self.get_model(i) for i in indices], indices

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if not self._fitted:
            to_return += "()"
            return to_return

        to_return += "("
        to_return += f"n_data={len(self.data_)}, "
        to_return += f"n_models={len(self.models_)}, "
        to_return += f"n_support={len(self.support_)}, "
        to_return += f"samples={len(self.samples_history)}"
        to_return += ")"

        return to_return


class PosteriorPiN(DistributionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise an warning of deprecation
        warnings.warn(
            "PosteriorPiN is deprecated. Use DistributionSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class DiscretePosteriorPiN(DiscreteDistributionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise an warning of deprecation
        warnings.warn(
            "DiscretePosteriorPiN is deprecated. Use DiscreteDistributionSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class ExplicitPosteriorPiN(ExplicitPosteriorSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise an warning of deprecation
        warnings.warn(
            "ExplicitPosteriorPiN is deprecated. Use ExplicitPosteriorSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
