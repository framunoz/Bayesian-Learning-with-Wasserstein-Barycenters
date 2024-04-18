"""
This module contains the classes that represent the distribution samplers. These classes are used to sample
distributions from a set of models, and they are divided into two main categories: discrete and continuous samplers.
"""
import abc
import collections as c
import typing as t
import warnings

import torch
import torch.nn as nn

import bwb.distributions as dist
import bwb.validation as validation
from bwb.config import config
from bwb.distributions.models import PDiscreteModelsSet, PDiscreteWeightedModelSet
from bwb.utils import _ArrayLike, _timeit_to_total_time, set_generator

__all__ = [
    "DistributionSampler",
    "DiscreteDistribSampler",
    "UniformDiscreteSampler",
    "ExplicitPosteriorSampler",
    "ContinuousDistribSampler",
    "GeneratorDistribSampler",
]

type device_t = str | torch.device | int | None


def _log_likelihood_default(model: dist.DiscreteDistribution, data: torch.Tensor):
    """Default log-likelihood of the posterior.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


class DistributionSampler[DistributionT](metaclass=abc.ABCMeta):
    r"""
    Base class for distributions that sampling other distributions. i.e. it represents a distribution
    :math:`\Lambda(dm) \in \mathcal{P}(\mathcal{M)}`, where :math:`\mathcal{M}` is the set of models.
    """

    def __init__(self) -> None:
        self.total_time = 0.0  # Total time to draw samples

    @abc.abstractmethod
    def draw(self, seed: t.Optional[int] = None) -> DistributionT:
        """Draw a sample."""
        ...

    @abc.abstractmethod
    def rvs(self, size: int = 1, seed: t.Optional[int] = None) -> t.Sequence[DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# noinspection PyAttributeOutsideInit
class DiscreteDistribSampler[DistributionT](DistributionSampler[DistributionT]):
    r"""
    Base class for distributions that have a discrete set of models. i.e. where the set of models is
    :math:`|\mathcal{M}| < +\infty`.

    As the support is discrete, the distribution can be represented as a vector of probabilities, and therefore, the
    sampling process is reduced to drawing an index from a multinomial distribution. This property allows to save the
    samples and the number of times each model has been sampled, to get statistics about the sampling process.
    """

    SAVE_SAMPLES = False

    def __init__(self, save_samples: bool = None) -> None:
        super().__init__()
        self.save_samples = save_samples or self.SAVE_SAMPLES
        self.samples_history: list[int] = []
        self.samples_counter: c.Counter[int] = c.Counter()
        self._models_cache: dict[int, DistributionT] = {}
        self._fitted = False

    @classmethod
    def set_save_samples(cls, save_samples: bool) -> None:
        """Set the attribute ``save_samples`` to the class."""
        if not isinstance(save_samples, bool):
            raise TypeError("The save_samples must be a boolean.")
        cls.SAVE_SAMPLES = save_samples

    def fit(self, models: PDiscreteModelsSet[DistributionT], *args, **kwargs):
        """Fit the distribution."""
        assert isinstance(models, PDiscreteModelsSet), (
            "The models must be a DiscreteModelsSet.\n"
            f"Missing methods: {set(dir(PDiscreteModelsSet)) - set(dir(models)) - {'_abc_impl', '_is_runtime_protocol',
                                                                                   '__abstractmethods__'} }"
        )

        self.models_: PDiscreteModelsSet[DistributionT] = models  # The set of models
        self.models_index_: torch.Tensor = torch.arange(
            len(models), device=config.device
        )  # The index of the models

        # The probabilities need to be set!
        return self

    def get_model(self, i: int) -> DistributionT:
        """Get the model with index i."""
        validation.check_is_fitted(self, ["models_"])
        if self._models_cache.get(i) is None:
            self._models_cache[i] = self.models_.get(i)
        return self._models_cache[i]

    def _draw(self, seed=None) -> tuple[DistributionT, int]:
        """To use template pattern on the draw method."""
        rng: torch.Generator = set_generator(seed=seed, device=config.device)

        i = torch.multinomial(
            input=self.probabilities_, num_samples=1, generator=rng
        ).item()
        i = int(i)

        return self.get_model(i), i

    @_timeit_to_total_time
    def draw(self, seed=None) -> DistributionT:
        """Draw a sample."""
        validation.check_is_fitted(self, ["models_", "probabilities_"])
        to_return, i = self._draw(seed)
        if self.save_samples:  # Register the sample
            self.samples_history.append(i)
            self.samples_counter[i] += 1
        return to_return

    def _rvs(
        self, size=1, seed=None
    ) -> tuple[t.Sequence[DistributionT], list[int]]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        rng: torch.Generator = set_generator(seed=seed, device=config.device)

        indices = torch.multinomial(
            input=self.probabilities_, num_samples=size, replacement=True, generator=rng
        )
        indices = indices.tolist()
        return [self.get_model(i) for i in indices], indices

    @_timeit_to_total_time
    def rvs(self, size=1, seed=None) -> t.Sequence[DistributionT]:
        """Samples as many distributions as the ``size`` parameter indicates."""
        validation.check_is_fitted(self, ["models_", "probabilities_"])
        to_return, list_indices = self._rvs(size, seed)
        if self.save_samples:  # Register the samples
            self.samples_history.extend(list_indices)
            self.samples_counter.update(list_indices)
        return to_return

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if self.save_samples:
            to_return += f"(samples={len(self.samples_history)})"

        return to_return


# noinspection PyAttributeOutsideInit
class UniformDiscreteSampler[DistributionT](DiscreteDistribSampler[DistributionT]):
    r"""
    A class representing a distribution sampler with a discrete set of models, and the probabilities are set to be
    uniform.

    This class inherits from the `DiscreteDistribSampler` class and provides methods to fit the sampler to a set of
    discrete models and generate samples from the fitted sampler.

    Attributes:
        probabilities_: A torch.Tensor representing the probabilities of each model in the sampler.
        support_: The indices of the models in the sampler.

    Methods:
        fit: Fits the sampler to a set of discrete models.
        __repr__: Returns a string representation of the sampler.

    """

    @_timeit_to_total_time
    def fit(self, models: PDiscreteModelsSet[DistributionT], *args, **kwargs):
        super().fit(models)
        self.probabilities_: torch.Tensor = torch.ones(
            len(models),
            device=config.device,
            dtype=config.dtype,
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


# noinspection PyAttributeOutsideInit
class ExplicitPosteriorSampler[DistributionT](DiscreteDistribSampler[DistributionT]):
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
        self,
        models: PDiscreteWeightedModelSet[DistributionT],
        data: _ArrayLike,
        **kwargs,
    ):
        r"""
        Fit the posterior distribution.

        :param data: The data to fit the posterior.
        :param models: The models to fit the posterior.
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


class ContinuousDistribSampler[DistributionT](DistributionSampler[DistributionT], abc.ABC):
    r"""
    Class for distributions that have a continuous set of models. i.e. where the set of models is
    :math:`|\mathcal{M}| = +\infty`.
    """

    ...


# noinspection PyAttributeOutsideInit
class GeneratorDistribSampler[DistributionT](ContinuousDistribSampler[DistributionT]):
    r"""
    Class for distributions that have a continuous set of models, and the models can be generated by a generator model.
    """

    SAVE_SAMPLES = False

    def __init__(self, save_samples: bool = None) -> None:
        """
        :param save_samples: If True, the samples are saved in the attribute ``samples_history``.
        """
        super().__init__()
        self.save_samples = save_samples or self.SAVE_SAMPLES
        self.samples_history: list[torch.Tensor] = []
        self._fitted = False

    def fit(
        self,
        generator: nn.Module,
        transform_out: t.Callable[[torch.Tensor], DistributionT],
        noise_sampler: t.Callable[[int], torch.Tensor],
    ):
        """
        Fits the distribution sampler.

        :param generator: The generator model.
        :type generator: nn.Module
        :param transform_out: A callable function that transforms the output of the generator.
        :type transform_out: Callable[[torch.Tensor], DistributionT]
        :param noise_sampler: A callable function that generates noise samples.
        :type noise_sampler: Callable[[int], torch.Tensor]
        :raises ValueError: If the noise sampler does not return a tensor.
        :raises ValueError: If the generator fails to generate a sample from the noise.
        :raises ValueError: If the transform_out fails to transform the output of the generator.
        :return: The fitted distribution sampler.
        :rtype: self
        """

        # Validation of the noise sampler
        if not callable(noise_sampler):
            raise TypeError("The noise sampler must be a callable.")
        try:
            noise = noise_sampler(42)
            if not isinstance(noise, torch.Tensor):
                raise ValueError(
                    "The noise sampler must return a tensor, but it returns a "
                    f"{type(noise)}"
                )
            if noise.shape[0] != 42:
                raise ValueError(
                    "The noise sampler must return a tensor of shape (42, ...), but it returns a "
                    f"tensor of shape {noise.shape}"
                )
        except:
            raise ValueError("The noise sampler must accept an integer.")
        self.noise_sampler_ = noise_sampler

        # Validation of the generator
        try:
            result = generator(noise)
        except:
            raise ValueError(
                "The generator must be able to generate a sample from the noise."
            )
        self.generator_ = generator

        # Validation of the transform_out
        try:
            noise = noise_sampler(1)
            result = generator(noise)
            result = transform_out(result)
        except:
            raise ValueError(
                "The transform_out must be able to transform the output of the generator."
            )
        self.transform_out_ = transform_out

        self._fitted = True
        return self

    def transform_noise(self, noise: torch.Tensor) -> DistributionT:
        """
        Transforms the given noise tensor using the generator network.

        :param noise: The noise tensor.
        :type noise: torch.Tensor
        :raises NotFittedError: If the distribution sampler is not fitted.
        :return: The transformed noise tensor.
        :rtype: DistributionT
        """
        validation.check_is_fitted(self, ["transform_out_", "generator_"])
        with torch.no_grad():
            gen_output = self.generator_(noise)
        return self.transform_out_(gen_output)
        # return self.transform_out_(self.generator_(noise))

    def _draw(self, seed=None) -> tuple[DistributionT, torch.Tensor]:
        noise: torch.Tensor = self.noise_sampler_(1)
        to_return: DistributionT = self.transform_noise(noise)
        return to_return, noise

    def draw(self, seed=None) -> DistributionT:
        """
        Draw a sample.

        :param seed: The seed to draw the sample.
        :type seed: int
        :raises NotFittedError: If the distribution sampler is not fitted.
        :return: The drawn sample.
        :rtype: DistributionT
        """
        validation.check_is_fitted(
            self, ["generator_", "transform_out_", "noise_sampler_"]
        )
        to_return, noise = self._draw(seed)
        if self.save_samples:
            self.samples_history.append(noise)
        return to_return

    def _rvs(
        self, size=1, seed=None
    ) -> tuple[t.Sequence[DistributionT], torch.Tensor]:
        noises: torch.Tensor = self.noise_sampler_(size)
        to_return: list[DistributionT] = [
            self.transform_noise(noise.unsqueeze(0)) for noise in noises
        ]
        return to_return, noises

    def rvs(self, size=1, seed=None) -> t.Sequence[DistributionT]:
        """
        Samples as many distributions as the `size` parameter indicates.

        :param size: The number of samples to draw.
        :type size: int
        :param seed: Seed for the random number generator. If None, a random seed will be used.
        :type seed: int, optional
        :raises NotFittedError: If the distribution sampler is not fitted.
        :return: A sequence of sampled distributions.
        :rtype: Sequence[DistributionT]
        """
        validation.check_is_fitted(
            self, ["generator_", "transform_out_", "noise_sampler_"]
        )
        to_return, noises = self._rvs(size, seed)
        if self.save_samples:
            self.samples_history.extend(noises)
        return to_return

    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if not self._fitted:
            to_return += "()"
            return to_return

        to_return += "("
        to_return += f"samples={len(self.samples_history)}"
        to_return += ")"
        return to_return


class PosteriorPiN(DistributionSampler, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

        # Raise a warning of deprecation
        warnings.warn(
            "PosteriorPiN is deprecated. Use DistributionSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class DiscretePosteriorPiN(DiscreteDistribSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise a warning of deprecation
        warnings.warn(
            "DiscretePosteriorPiN is deprecated. Use DiscreteDistributionSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


class ExplicitPosteriorPiN(ExplicitPosteriorSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise a warning of deprecation
        warnings.warn(
            "ExplicitPosteriorPiN is deprecated. Use ExplicitPosteriorSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
