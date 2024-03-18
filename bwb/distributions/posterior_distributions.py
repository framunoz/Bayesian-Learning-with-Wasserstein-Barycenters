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
    "DiscreteDistribSampler",
    "UniformDiscreteSampler",
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

    # NOTE: Esta parte igual está rara. No se debería de dejar que esta clase sepa cómo calcular las probabilidades, sólo debería de contener los modelos y ya.
    def compute_likelihood(self, data: _ArrayLike = None, **kwargs) -> torch.Tensor:
        """
        Compute the probabilities of the data given the models.

        :param data: The data to compute the probabilities.
        :return: A tensor with the probabilities.
        """
        ...

    def get(self, i: int, **kwargs) -> _DistributionT:
        """Get the model at the index ``i``."""
        ...

    def __len__(self) -> int:
        """Get the number of models."""
        ...


class DistributionSampler(abc.ABC, t.Generic[_DistributionT]):
    r"""
    Base class for distributions that sampling other distributions. i.e. it represents a distribution :math:`\Lambda(dm) \in \mathcal{P}(\mathcal{M)}`, where :math:`\mathcal{M}` is the set of models.
    """

    def __init__(self) -> None:
        self.total_time = 0.0  # Total time to draw samples

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


class DiscreteDistribSampler(DistributionSampler[_DistributionT]):
    r"""
    Base class for distributions that have a discrete set of models. i.e. where the set of models is :math:`|\mathcal{M}| < +\infty`.

    As the support is discrete, the distribution can be represented as a vector of probabilities, and therefore, the sampling process is reduced to drawing an index from a multinomial distribution. This property allows to save the samples and the number of times each model has been sampled, to get statistics about the sampling process.
    """

    def __init__(self, save_samples: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_samples = save_samples
        self.samples_history: list[int] = []
        self.samples_counter: c.Counter[int] = c.Counter()
        self._models_cache: dict[int, _DistributionT] = {}
        self._fitted = False

    def fit(self, models: DiscreteModelsSet[_DistributionT], *args, **kwargs):
        """Fit the distribution."""
        assert isinstance(models, DiscreteModelsSet), (
            "The models must be a DiscreteModelsSet.\n"
            f"Missing methods: {set(dir(DiscreteModelsSet)) - set(dir(models)) - {'_abc_impl', '_is_runtime_protocol', '__abstractmethods__'}}"
        )

        self.models_: DiscreteModelsSet[_DistributionT] = models  # The set of models
        self.models_index_: torch.Tensor = torch.arange(
            len(models), device=config.device
        )  # The index of the models

        # The probabilities needs to be set!
        return self

    def _draw(self, seed=None, *args, **kwargs) -> tuple[_DistributionT, int]:
        """To use template pattern on the draw method."""
        rng: torch.Generator = _set_generator(seed=seed, device=config.device)

        i = torch.multinomial(
            input=self.probabilities_, num_samples=1, generator=rng
        ).item()
        i = int(i)

        return self.get_model(i), i

    @_timeit_to_total_time
    def draw(self, seed=None, *args, **kwargs) -> _DistributionT:
        """Draw a sample."""
        validation.check_is_fitted(self, ["models_", "probabilities_"])
        to_return, i = self._draw(seed, *args, **kwargs)
        if self.save_samples:  # Register the sample
            self.samples_history.append(i)
            self.samples_counter[i] += 1
        return to_return

    def _rvs(
        self, size=1, seed=None, *args, **kwargs
    ) -> tuple[t.Sequence[_DistributionT], list[int]]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        rng: torch.Generator = _set_generator(seed=seed, device=config.device)

        indices = torch.multinomial(
            input=self.probabilities_, num_samples=size, replacement=True, generator=rng
        )
        indices = indices.tolist()
        return [self.get_model(i) for i in indices], indices

    @_timeit_to_total_time
    def rvs(self, size=1, seed=None, *args, **kwargs) -> t.Sequence[_DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        validation.check_is_fitted(self, ["models_", "probabilities_"])
        to_return, list_indices = self._rvs(size, seed, *args, **kwargs)
        if self.save_samples:  # Register the samples
            self.samples_history.extend(list_indices)
            self.samples_counter.update(list_indices)
        return to_return

    def get_model(self, i: int) -> _DistributionT:
        """Get the model with index i."""
        validation.check_is_fitted(self, ["models_"])
        if self._models_cache.get(i) is None:
            self._models_cache[i] = self.models_.get(i)
        return self._models_cache[i]

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if self.save_samples:
            to_return += f"(samples={len(self.samples_history)})"

        return to_return


class UniformDiscreteSampler(DiscreteDistribSampler[_DistributionT]):
    r""" """

    @_timeit_to_total_time
    def fit(self, models: DiscreteModelsSet[_DistributionT], *args, **kwargs):
        super().fit(models)
        self.probabilities_: torch.Tensor = torch.ones(
            len(models), device=config.device
        ) / len(models)

        self.support_ = self.models_index_

        self._fitted = True

        return self

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if not self._fitted:
            to_return += "()"
            return to_return

        to_return += "("
        to_return += f"n_models={len(self.models_)}, "
        to_return += f"samples={len(self.samples_history)}"
        to_return += ")"

        return to_return


class ExplicitPosteriorSampler(DiscreteDistribSampler[_DistributionT]):
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

    @_timeit_to_total_time
    def fit(
        self, models: DiscreteModelsSet[_DistributionT], data: _ArrayLike, **kwargs
    ):
        """
        Fit the posterior distribution.

        :param data: The data to fit the posterior.
        :param models: The models to fit the posterior.
        :param batch_size: The batch size to compute the probabilities.
        :return: The fitted posterior.
        """
        super().fit(models)
        self.data_: torch.Tensor = torch.as_tensor(data, device=config.device)

        data = self.data_.reshape(1, -1)

        self.probabilities_: torch.Tensor = models.compute_likelihood(data, **kwargs)

        self.support_ = self.models_index_[self.probabilities_ > config.eps]

        self._fitted = True

        return self

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


class ContinuousDistribSampler(DistributionSampler[_DistributionT]): ...


class PosteriorPiN(DistributionSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise an warning of deprecation
        warnings.warn(
            "PosteriorPiN is deprecated. Use DistributionSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class DiscretePosteriorPiN(DiscreteDistribSampler):
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
