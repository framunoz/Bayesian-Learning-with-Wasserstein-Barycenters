import copy
import random
import typing as t
import warnings

import hamiltorch
import torch
from hamiltorch import Sampler

from bwb import distributions as dist
from bwb.config import config
from bwb.distributions import DiscreteDistribSampler
from bwb.distributions.distribution_samplers import BaseGeneratorDistribSampler, GeneratorP, seed_t
from bwb.distributions.models import DiscreteWeightedModelSetP
from bwb.utils import array_like_t, integrated_time, timeit_to_total_time
from bwb.validation import check_is_fitted

__all__ = [
    "ExplicitPosteriorSampler",
    "BaseLatentMCMCPosteriorSampler",
    "LatentMCMCPosteriorSampler",
    "NUTSPosteriorSampler",
]


def _log_likelihood_default(model: dist.DiscreteDistribution, data: torch.Tensor) -> torch.Tensor:
    """Default log-likelihood of the posterior.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


def log_prior(z: torch.Tensor) -> torch.Tensor:
    """
    Corresponds to the log prior of a Normal(z; 0, 1) distribution.

    :param z: The tensor to evaluate the log prior. Also represents the noise.
    :return: The log prior of the tensor.
    """
    z = z.squeeze()
    z_2 = z ** 2
    return -0.5 * torch.sum(z_2)  # -0.5 * \sum_{i=1}^d z_i^2


def log_likelihood_latent(
    z: torch.Tensor,
    data: torch.Tensor,
    generator: GeneratorP,
    transform_out: t.Callable[[torch.Tensor], torch.Tensor],
):
    """
    Log-likelihood of the latent variable z, given the data, the generator, and the transformation.

    :param z: The latent variable. Also represents the noise.
    :param data: The data to evaluate the log-likelihood.
    :param generator: The generator to evaluate the latent variable.
    :param transform_out: The transformation to apply to the generator output.
    :return: The log-likelihood of the latent variable.
    """
    eps = torch.finfo(z.dtype).eps

    # A model is the forward pass of the generator and transformed
    z = torch.reshape(z, (1, -1, 1, 1))
    with torch.no_grad():
        m = generator(z)
    m = transform_out(m)
    m = m.reshape((-1,))

    m_data = m.take(data)  # Take m(x_i) for all x_i
    m_data = m_data + eps  # to avoid log(0)
    logits = torch.log(m_data)  # log m(x_i)

    return torch.sum(logits)  # \sum_{i=1}^n \log m(x_i)


def log_posterior(
    z: torch.Tensor,
    data: torch.Tensor,
    generator: GeneratorP,
    transform_out: t.Callable[[torch.Tensor], torch.Tensor],
):
    """
    Log-posterior of the latent variable z, given the data, the generator, and the transformation.

    :param z: The latent variable. Also represents the noise.
    :param data: The data to evaluate the log-likelihood.
    :param generator: The generator to evaluate the latent variable.
    :param transform_out: The transformation to apply to the generator output.
    :return: The log-posterior of the latent variable.
    """
    return log_prior(z) + log_likelihood_latent(z, data, generator, transform_out)


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

    @timeit_to_total_time
    def fit(
        self,
        models: DiscreteWeightedModelSetP[DistributionT],
        data: array_like_t,
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


# noinspection PyAttributeOutsideInit
class BaseLatentMCMCPosteriorSampler(BaseGeneratorDistribSampler[dist.DistributionDraw]):
    """
    Base class for MCMC posterior samplers.
    """

    def __init__(
        self,
        log_prob_fn=log_posterior,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        sampler: Sampler = Sampler.HMC,
        save_samples: bool = True,
        **kwargs,
    ):
        super().__init__(save_samples=save_samples)
        self.log_prob_fn = log_prob_fn
        self.samples_history: list[list[torch.Tensor]] = [[]]
        self.samples_cache: list[torch.Tensor] = []
        self._hamiltorch_kwargs = dict(
            num_samples=num_samples,
            num_steps_per_sample=num_steps_per_sample,
            burn=burn,
            step_size=step_size,
            sampler=sampler,
        )
        self._hamiltorch_kwargs.update(kwargs)
        self._hamiltorch_kwargs.setdefault("debug", 0)
        self.log = None

    # noinspection PyMethodOverriding
    @t.override
    def fit(
        self,
        generator: GeneratorP,
        transform_out: t.Callable[[torch.Tensor], torch.Tensor],
        noise_sampler: t.Callable[[int], torch.Tensor],
        data: torch.Tensor,
    ) -> t.Self:
        super().fit(generator=generator, transform_out=transform_out, noise_sampler=noise_sampler)

        if not isinstance(data, torch.Tensor):
            raise ValueError("data must be a torch.Tensor")
        self.data_ = data

        return self

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of the latent variable z.

        :param z: The latent variable. Also represents the noise.
        :return: The log-probability of the latent variable.
        """
        return self.log_prob_fn(z, self.data_, self.generator_, self.transform_out_)

    def reset_samples(self) -> t.Self:
        """
        Reset the samples cache.

        :return: The object itself.
        """
        self.samples_cache = []
        return self

    def _get_hamiltorch_kwargs(self, **kwargs) -> dict:
        hamiltorch_kwargs = self._hamiltorch_kwargs.copy()
        specific_kwargs: dict[str, t.Any] = dict(log_prob_func=self.log_prob, **kwargs)
        hamiltorch_kwargs.update([(k, v) for k, v in specific_kwargs.items() if v is not None])
        hamiltorch_kwargs["num_samples"] = hamiltorch_kwargs["num_samples"] + hamiltorch_kwargs["burn"]

        return hamiltorch_kwargs

    @timeit_to_total_time
    def run(
        self,
        params_init: torch.Tensor = None,
        num_samples: int = None,
        burn: int = None,
        **kwargs,
    ) -> t.Self:
        """
        Run the MCMC sampler.

        :param params_init: The initial point to start the sampler.
        :param num_samples: The number of samples to generate.
        :param burn: The number of burn-in samples to generate.
        :param kwargs: Additional arguments to pass to the function `hamiltorch.sample`.
        :return: The object itself.
        """
        check_is_fitted(self, ["generator_", "transform_out_", "noise_sampler_", "data_"])
        z_init: torch.Tensor = params_init if params_init is not None else self.noise_sampler_(1).squeeze()

        hamiltorch_kwargs = self._get_hamiltorch_kwargs(
            params_init=z_init,
            num_samples=num_samples,
            burn=burn,
            **kwargs
        )

        debug = hamiltorch_kwargs["debug"]

        if debug == 2:
            samples, self.log = hamiltorch.sample(**hamiltorch_kwargs)
        else:
            samples = hamiltorch.sample(**hamiltorch_kwargs)

        self.samples_history[0].extend(samples)

        return self

    def get_chain(self, flat: bool = False, thin: int = 1, discard: int = 0) -> torch.Tensor:
        """
        Returns the chain with shape (n_step, n_walker, n_param), or (n_step * n_walker, n_param) if flat=True.

        :param flat: Whether to return the chain flattened.
        :param thin: Take every thin sample.
        :param discard: Discard the first discard samples.
        :return: The chain.
        """
        to_return = torch.stack(self.samples_history[0][discard::thin]).unsqueeze(1)
        if flat:
            to_return = to_return.reshape(-1, to_return.shape[-1])
        return to_return

    def get_autocorr_time(self, thin=1, discard=0, **kwargs) -> torch.Tensor:
        """
        Returns the autocorrelation time of the chain.

        :param thin: Take every thin sample.
        :param discard: Discard the first discard samples.
        :param kwargs: Additional arguments to pass to the function `integrated_time`.
        :return: The autocorrelation time.
        """
        chain = self.get_chain(thin=thin, discard=discard)
        return thin * integrated_time(chain, **kwargs)

    def shuffle_samples_cache(self, thin: int = 1, discard: int = 0) -> t.Self:
        """
        Shuffle the samples cache.

        :param thin: Take every thin sample.
        :param discard: Discard the first discard samples.
        :return: The object itself.
        """
        to_extend = self.get_chain(flat=True, thin=thin, discard=discard)
        to_extend = [sample for sample in to_extend]
        to_extend = random.sample(to_extend, len(to_extend))
        self.samples_cache.extend(to_extend)
        return self

    @t.final
    @t.override
    def _draw(self, seed: seed_t = None) -> tuple[dist.DistributionDraw, torch.Tensor]:
        if not self.samples_cache:
            self.run()
            self.shuffle_samples_cache()

        z_sampled = self.samples_cache.pop(0).reshape(1, -1, 1, 1)
        distribution: dist.DistributionDraw = self.transform_noise(z_sampled)
        return distribution, z_sampled

    @t.final
    @t.override
    def _rvs(self, size: int = 1, seed: seed_t = None) -> tuple[t.Sequence[dist.DistributionDraw], torch.Tensor]:
        if len(self.samples_cache) < size:
            self.run(num_samples=size * 50)
            self.shuffle_samples_cache()

        samples, self.samples_cache = self.samples_cache[:size], self.samples_cache[size:]
        samples = [sample.reshape(1, -1, 1, 1) for sample in samples]

        samples_: list[dist.DistributionDraw] = []
        for z_ in samples:
            distribution: dist.DistributionDraw = self.transform_noise(z_)
            samples_.append(distribution)

        return samples_, samples

    def draw(self, seed: seed_t = None) -> dist.DistributionDraw:
        return self._draw(seed)[0]

    def rvs(self, size: int = 1, seed: seed_t = None) -> t.Sequence[dist.DistributionDraw]:
        return self._rvs(size, seed)[0]

    @t.final
    @t.override
    def create_distribution(self, input: torch.Tensor) -> dist.DistributionDraw:
        return dist.DistributionDraw.from_grayscale_weights(input.squeeze())

    def _additional_repr_(self, sep) -> str:
        to_return = ""

        if hasattr(self, "data_"):
            to_return += f"n_data={len(self.data_)}" + sep

        if self.samples_cache:
            to_return += f"n_cached_samples={len(self.samples_cache)}" + sep

        if self.samples_history[0]:
            to_return += f"n_total_iterations={len(self.samples_history[0])}" + sep

        return to_return

    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)
        new = self.__class__(
            log_prob_fn=self.log_prob_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )
        new.samples_history = copy.copy(self.samples_history)
        new.samples_cache = copy.copy(self.samples_cache)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        hamiltorch_kwargs = copy.deepcopy(self._hamiltorch_kwargs, memo)

        new = self.__class__(
            log_prob_fn=self.log_prob_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )
        self.samples_history = copy.deepcopy(self.samples_history, memo)
        self.samples_cache = copy.deepcopy(self.samples_cache, memo)
        self.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


class LatentMCMCPosteriorSampler(BaseLatentMCMCPosteriorSampler):
    def __init__(
        self,
        n_workers: int = None,
        n_walkers: int = 1,
        log_prob_fn=log_posterior,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        sampler: Sampler = Sampler.HMC,
        parallel: bool = False,
        save_samples: bool = True,
        **kwargs,
    ):
        super().__init__(
            log_prob_fn=log_prob_fn,
            num_samples=num_samples,
            num_steps_per_sample=num_steps_per_sample,
            burn=burn,
            step_size=step_size,
            sampler=sampler,
            save_samples=save_samples,
            **kwargs
        )
        self.n_workers = n_workers
        self.n_walkers = n_walkers
        self.parallel = parallel
        self.samples_history: list[list[torch.Tensor]] = [[] for _ in range(n_walkers)]

    @timeit_to_total_time
    def run(
        self,
        n_workers: int = None,
        n_walkers: int = None,
        seeds: t.Sequence[seed_t] = None,
        num_samples: int = None,
        burn: int = None,
        parallel: bool = None,
        **kwargs,
    ) -> t.Self:
        check_is_fitted(self, ["generator_", "transform_out_", "noise_sampler_", "data_"])

        n_workers = n_workers if n_workers is not None else self.n_workers
        n_walkers = n_walkers if n_walkers is not None else self.n_walkers
        parallel = parallel if parallel is not None else self.parallel

        seeds = seeds if seeds is not None else torch.arange(n_walkers)
        assert len(seeds) == n_walkers, "The number of seeds must match the number of walkers."

        def _prior(): return self.noise_sampler_(1).squeeze()

        hamiltorch_kwargs = self._get_hamiltorch_kwargs(num_samples=num_samples, burn=burn, **kwargs)
        debug = hamiltorch_kwargs["debug"]

        chain = hamiltorch.util.setup_chain(hamiltorch.sample, _prior, hamiltorch_kwargs)

        params_hmc = hamiltorch.util.multi_chain(chain, n_workers, seeds, parallel)

        other_list = []
        for i, params in enumerate(params_hmc):
            if debug == 2:
                params, other = params
                self.samples_history[i].extend(params)
                other_list.append(other)
            self.samples_history[i].extend(params)

        self.log = other_list

        return self

    def get_chain(self, flat: bool = False, thin: int = 1, discard: int = 0) -> torch.Tensor:
        samples_per_chain = [torch.stack(samples) for samples in self.samples_history]
        to_return = torch.stack(samples_per_chain, dim=1)
        to_return = to_return[discard::thin]
        if flat:
            to_return = to_return.flatten(0, 1)
        return to_return

    def _additional_repr_(self, sep) -> str:
        to_return = super()._additional_repr_(sep)

        if self.n_workers is not None:
            to_return += f"n_workers={self.n_workers}" + sep

        if self.n_walkers > 1:
            to_return += f"n_walkers={self.n_walkers}" + sep

        if self.parallel:
            to_return += "parallel=True" + sep

        return to_return

    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)

        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_prob_fn=self.log_prob_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )
        new.samples_history = copy.copy(self.samples_history)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        hamiltorch_kwargs = copy.deepcopy(self._hamiltorch_kwargs, memo)

        # noinspection DuplicatedCode
        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_prob_fn=self.log_prob_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )
        new.samples_history = copy.deepcopy(self.samples_history, memo)
        new.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


class NUTSPosteriorSampler(LatentMCMCPosteriorSampler):
    def __init__(
        self,
        log_prob_fn=log_posterior,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        desired_accept_rate: float = 0.6,
        n_workers: int = None,
        n_walkers: int = 1,
        parallel: bool = False,
        save_samples: bool = True,
        **kwargs,
    ):
        super().__init__(
            log_prob_fn=log_prob_fn,
            num_samples=num_samples,
            num_steps_per_sample=num_steps_per_sample,
            burn=burn,
            step_size=step_size,
            sampler=Sampler.HMC_NUTS,
            desired_accept_rate=desired_accept_rate,
            n_workers=n_workers,
            n_walkers=n_walkers,
            parallel=parallel,
            save_samples=save_samples,
            **kwargs
        )

    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)

        new = self.__class__(
            log_prob_fn=self.log_prob_fn,
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            desired_accept_rate=hamiltorch_kwargs.pop("desired_accept_rate"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )

        new.samples_history = copy.copy(self.samples_history)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    # noinspection DuplicatedCode
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        hamiltorch_kwargs = copy.deepcopy(self._hamiltorch_kwargs, memo)

        new = self.__class__(
            log_prob_fn=self.log_prob_fn,
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            desired_accept_rate=hamiltorch_kwargs.pop("desired_accept_rate"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            **hamiltorch_kwargs
        )

        new.samples_history = copy.deepcopy(self.samples_history, memo)
        new.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


class ExplicitPosteriorPiN(ExplicitPosteriorSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise a warning of deprecation
        warnings.warn(
            "ExplicitPosteriorPiN is deprecated. Use ExplicitPosteriorSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )


def main():
    posterior = BaseLatentMCMCPosteriorSampler()
    print(posterior)
    print(posterior.samples_history)

    print(LatentMCMCPosteriorSampler(parallel=True))
    print(LatentMCMCPosteriorSampler(n_walkers=2))
    print(LatentMCMCPosteriorSampler(n_workers=2, parallel=True))


if __name__ == '__main__':
    main()
