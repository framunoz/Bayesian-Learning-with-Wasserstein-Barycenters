import copy
import random
import warnings
from typing import Any, Callable, final, override, Protocol, Self, Sequence

import hamiltorch
import torch
from hamiltorch import Sampler

import bwb.logging_ as logging
from bwb import distributions as D
from bwb.config import conf
from bwb.distributions import DiscreteDistribSampler
from bwb.distributions.distribution_samplers import (
    BaseGeneratorDistribSampler,
    GeneratorP,
)
from bwb.distributions.models import DiscreteWeightedModelSetP
from bwb.utils import (
    ArrayLikeT,
    check_is_fitted,
    integrated_time,
    SeedT,
    timeit_to_total_time,
)

_log = logging.get_logger(__name__)

__all__ = [
    "DistributionP",
    "ExplicitPosteriorSampler",
    "BaseLatentMCMCPosteriorSampler",
    "LatentMCMCPosteriorSampler",
    "NUTSPosteriorSampler",
]


class DistributionP(Protocol):
    """
    Protocol class for distributions.
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of the distribution.
        """


def log_likelihood_model(
    model: DistributionP,
    data: torch.Tensor
) -> torch.Tensor:
    """Log-likelihood of a model.

    :param model: A model to obtain its log-likelihood
    :param data: The data to evaluate in the model
    :return: The log-likelihood as a torch tensor
    """
    return torch.sum(model.log_prob(data))


def likelihood_model(
    model: DistributionP,
    data: torch.Tensor
) -> torch.Tensor:
    """Likelihood of a model.

    :param model: A model to obtain its likelihood
    :param data: The data to evaluate in the model
    :return: The likelihood as a torch tensor
    """
    return torch.exp(log_likelihood_model(model, data))


def log_prior(z: torch.Tensor) -> torch.Tensor:
    """
    Corresponds to the log prior of a Normal(z; 0, 1) distribution.

    :param z: The tensor to evaluate the log prior. Also represents the
        noise.
    :return: The log prior of the tensor.
    """
    z = z.squeeze()
    z_2 = z ** 2
    return -0.5 * torch.sum(z_2)  # -0.5 * \sum_{i=1}^d z_i^2


def log_likelihood_latent(
    z: torch.Tensor,
    data: torch.Tensor,
    generator: GeneratorP,
    transform_out: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Log-likelihood of the latent variable z, given the data, the
    generator, and the transformation.

    :param z: The latent variable. Also represents the noise.
    :param data: The data to evaluate the log-likelihood.
    :param generator: The generator to evaluate the latent variable.
    :param transform_out: The transformation to apply to the generator
        output.
    :return: The log-likelihood of the latent variable.
    """
    eps = conf.eps

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
    transform_out: Callable[[torch.Tensor], torch.Tensor],
):
    """
    Log-posterior of the latent variable z, given the data, the
    generator, and the transformation.

    :param z: The latent variable. Also represents the noise.
    :param data: The data to evaluate the log-likelihood.
    :param generator: The generator to evaluate the latent variable.
    :param transform_out: The transformation to apply to the generator
        output.
    :return: The log-posterior of the latent variable.
    """
    return log_prior(z) + log_likelihood_latent(z, data, generator,
                                                transform_out)


# noinspection PyAttributeOutsideInit
class ExplicitPosteriorSampler[DistributionT](
    DiscreteDistribSampler[DistributionT]
):
    r"""Distribution that uses the strategy of calculating all
    likelihoods by brute force. This class implements likelihoods of
    the form

    .. math::
        \mathcal{L}_n(m) = \prod_{i=1}^{n} \rho_{m}(x_i)

    using the log-likelihood for stability. Finally, to compute the
    sampling probabilities, for a discrete set :math:`\mathcal{M}` of
    models, using a uniform prior, we have the posterior explicit by

    .. math::
        \Pi_n(m) = \frac{\mathcal{L}_n(m)}{\sum_{\bar m \in \mathcal{M}}
         \mathcal{L}_n(\bar m)}

    """

    @timeit_to_total_time
    def fit(
        self,
        models: DiscreteWeightedModelSetP[DistributionT],
        data: ArrayLikeT,
        **kwargs,
    ) -> Self:
        r"""
        Fit the posterior distribution.

        :param data: The data to fit the posterior.
        :param models: The models to fit the posterior.
        :return: The fitted posterior.
        """
        super().fit(models)
        self.data_: torch.Tensor = torch.as_tensor(data, device=conf.device)

        data = self.data_.reshape(1, -1)

        self.probabilities_: torch.Tensor = models.compute_likelihood(
            data,
            **kwargs
        )

        self.support_ = self.models_index_[self.probabilities_ > conf.eps]

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
class BaseLatentMCMCPosteriorSampler(
    BaseGeneratorDistribSampler[D.DistributionDraw]
):
    """
    Base class for MCMC posterior samplers. This class try to mirror
    the `emcee` package, but using the ``hamiltorch`` package as a
    backend. For this reason, many of the methods could be similar.
    """

    def __init__(
        self,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        sampler: Sampler = Sampler.HMC,
        save_samples: bool = True,
        log_likelihood_fn=log_likelihood_latent,
        log_prior_fn=log_prior,
        use_half: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(save_samples=save_samples, use_half=use_half)
        # The log-likelihood and log-prior functions
        self.log_likelihood_fn = log_likelihood_fn
        self.log_prior_fn = log_prior_fn

        # The history of steps
        self.n_walkers: int = 1
        self.n_steps: int = 0
        self.chains: list[list[torch.Tensor]] = self._create_empty_chains(1)

        # The cache for the samples
        self.samples_cache: list[torch.Tensor] = []

        # Arguments for the hamiltorch.sample function
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

        # The autocorrelation time
        self._mean_autocorr_time = None  # not set yet
        self._autocorr_time = None  # not set yet
        self._n_effective = None  # not set yet

    def __len__(self) -> int:
        """
        The number of steps in every chain.
        """
        return sum(len(chain) for chain in self.chains)

    def _create_empty_chains(
        self,
        n_walkers: int = None
    ) -> list[list[torch.Tensor]]:
        """
        Create an empty list of chains.

        :param n_walkers: The number of walkers.
        :return: The list of chains.
        """
        return [[] for _ in range(n_walkers or self.n_walkers)]

    def reset_chain(self) -> Self:
        """
        Reset the chain.

        :return: The object itself.
        """
        self.chains = self._create_empty_chains()
        self.n_steps = 0
        self._mean_autocorr_time = None
        self._autocorr_time = None
        self._n_effective = None
        return self

    def reset_samples(self) -> Self:
        """
        Reset the samples cache.

        :return: The object itself.
        """
        self.samples_cache = []
        return self

    # noinspection PyMethodOverriding
    @override
    def fit(
        self,
        generator: GeneratorP,
        transform_out: Callable[[torch.Tensor], torch.Tensor],
        noise_sampler: Callable[[int], torch.Tensor],
        data: torch.Tensor,
        **kwargs,
    ) -> Self:
        super().fit(
            generator=generator,
            transform_out=transform_out,
            noise_sampler=noise_sampler
        )

        if not isinstance(data, torch.Tensor):
            raise ValueError("data must be a torch.Tensor")
        self.data_ = data

        return self

    def log_likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-likelihood of the latent variable z.

        :param z: The latent variable. Also represents the noise.
        :return: The log-likelihood of the latent variable.
        """
        if not self._fitted:
            check_is_fitted(self,
                            ["generator_", "transform_out_", "data_"])
        return self.log_likelihood_fn(z, self.data_, self.generator_,
                                      self.transform_out_)

    def log_prior(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-prior of the latent variable z.

        :param z: The latent variable. Also represents the noise.
        :return: The log-prior of the latent variable.
        """
        return self.log_prior_fn(z)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of the latent variable z.

        :param z: The latent variable. Also represents the noise.
        :return: The log-probability of the latent variable.
        """
        return self.log_likelihood(z) + self.log_prior(z)

    def _get_hamiltorch_kwargs(self, **kwargs) -> dict:
        hamiltorch_kwargs = self._hamiltorch_kwargs.copy()
        specific_kwargs: dict[str, Any] = dict(log_prob_func=self.log_prob,
                                               **kwargs)
        hamiltorch_kwargs.update(
            [(k, v) for k, v in specific_kwargs.items() if v is not None]
        )
        hamiltorch_kwargs["num_samples"] = (hamiltorch_kwargs["num_samples"]
                                            + hamiltorch_kwargs["burn"])

        return hamiltorch_kwargs

    @timeit_to_total_time
    def run(
        self,
        initial_state: torch.Tensor = None,
        n_steps: int = None,
        burn: int = None,
        seed: SeedT = None,
        **kwargs,
    ) -> Self:
        """
        Run the MCMC sampler.

        :param initial_state: The initial point to start the sampler.
        :param n_steps: The number of samples to generate.
        :param burn: The number of burn-in samples to generate.
        :param seed: The seed to set the random number generator.
        :param kwargs: Additional arguments to pass to the function
            `hamiltorch.sample`.
        :return: The object itself.
        """
        check_is_fitted(self, ["generator_", "transform_out_",
                               "noise_sampler_", "data_"])
        if initial_state is None:
            z_init = self._get_noise(1, seed)
        else:
            z_init = initial_state
        z_init = z_init.squeeze()

        hamiltorch_kwargs = self._get_hamiltorch_kwargs(
            params_init=z_init,
            num_samples=n_steps,
            burn=burn,
            **kwargs
        )

        debug = hamiltorch_kwargs["debug"]

        if debug == 2:
            chain, self.log = hamiltorch.sample(**hamiltorch_kwargs)
        else:
            chain = hamiltorch.sample(**hamiltorch_kwargs)
        chain = self._set_half_dtype(chain)

        self.chains[0].extend(chain)
        self.n_steps += len(chain)

        return self

    def get_chain(
        self,
        flat: bool = False,
        thin: int = 1,
        discard: int = 0
    ) -> torch.Tensor:
        """
        This method is a copy of ``emcee``. Get the stored chain of MCMC
        sample. Returns the chain with shape  ``(n_step, n_walker,
        n_param)``, or ``(n_step * n_walker, n_param)`` if ``flat=True``.

        :param flat: Flatten the chain across the ensemble.
            (default: ``False``)
        :param thin: Take only every ``thin`` steps from the chain.
            (default: ``1``)
        :param discard: Discard the first ``discard`` steps in the chain
            as burn-in. (default: ``0``)
        :return: The MCMC samples
        """
        to_return = torch.stack(self.chains[0][discard::thin]).unsqueeze(1)
        if flat:
            to_return = to_return.reshape(-1, to_return.shape[-1])
        to_return = self._set_normal_dtype(to_return)
        return to_return

    @property
    def mean_autocorr_time(self) -> torch.Tensor:
        """
        Get the autocorrelation time of the chain.

        :return: The autocorrelation time of the chain.
        """
        if self._mean_autocorr_time is None:
            logging.raise_warning(
                "The autocorrelation time is not set yet.",
                _log, RuntimeWarning, stacklevel=2
            )
            return torch.tensor(0.0,
                                dtype=self.dtype, device=self.device)
        return self._mean_autocorr_time

    @property
    def autocorr_time(self) -> torch.Tensor:
        """
        Get the autocorrelation time of the chain.

        :return: The autocorrelation time of the chain.
        """
        if self._autocorr_time is None:
            logging.raise_warning(
                "The autocorrelation time is not set yet.",
                _log, RuntimeWarning, stacklevel=2
            )
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return self._autocorr_time

    @property
    def n_effective(self) -> int:
        """
        Get the effective number of samples.
        """
        if self._n_effective is None:
            logging.raise_warning(
                "The effective number of samples is not set yet.",
                _log, RuntimeWarning, stacklevel=2
            )
            return 0
        return self._n_effective

    @property
    def n_cached_samples(self) -> int:
        """
        Get the number of cached samples.
        """
        return len(self.samples_cache)

    def get_autocorr_time(self, thin=1, discard=0, **kwargs) -> torch.Tensor:
        """
        This method is a copy of ``emcee``. Compute an estimate of the
        autocorrelation time for each parameter.

        :param thin: Use only every ``thin`` steps from the
                chain. The returned estimate is multiplied by ``thin`` so the
                estimated time is in units of steps, not thinned steps.
                (default: ``1``)
        :param discard: Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)
        :param kwargs: Additional arguments to pass to the function
            :func:`bwb.utils.integrated_time`.
        :return: The integrated autocorrelation time estimate for the
                chain for each parameter.
        """
        chain = self.get_chain(thin=thin, discard=discard)
        autocorr_time = thin * integrated_time(chain, **kwargs)
        self._autocorr_time = autocorr_time
        self._mean_autocorr_time = torch.mean(autocorr_time)
        self._n_effective = len(self) / self.mean_autocorr_time
        return autocorr_time

    def shuffle_samples_cache(
        self,
        thin: int = None,
        discard: int = 0
    ) -> Self:
        """
        Shuffle the samples cache to avoid correlation between samples.

        :param thin: Take only every ``thin`` steps from the chain.
            (default: ``1``)
        :param discard: Discard the first ``discard`` steps in the
            chain as burn-in. (default: ``0``)
        :return: The object itself.
        """
        if thin is None:
            if self._mean_autocorr_time is not None:
                thin = int(self.mean_autocorr_time.item())
            else:
                thin = 1
        to_extend = self.get_chain(flat=True, thin=thin, discard=discard)
        to_extend = [sample for sample in to_extend]
        to_extend = random.sample(to_extend, len(to_extend))
        to_extend = self._set_half_dtype(to_extend)

        self.samples_cache.extend(to_extend)
        return self

    @final
    @override
    def _draw(
        self,
        seed: SeedT = None
    ) -> tuple[D.DistributionDraw, torch.Tensor]:
        if not self.samples_cache:
            self.run(n_steps=50)
            self.shuffle_samples_cache()

        sample = self.samples_cache.pop(0)
        sample = self._set_normal_dtype(sample)

        z_sampled = sample.reshape(1, -1, 1, 1)
        distribution: D.DistributionDraw = self.transform_noise(z_sampled)
        return distribution, z_sampled

    @final
    @override
    def _sample(
        self,
        size: int = 1,
        seed: SeedT = None
    ) -> tuple[Sequence[D.DistributionDraw], Sequence[torch.Tensor]]:
        if len(self.samples_cache) < size:
            self.run(n_steps=size * 50)
            self.shuffle_samples_cache()

        samples, self.samples_cache = self.samples_cache[
                                      :size], self.samples_cache[size:]
        samples = self._set_normal_dtype(samples)
        samples = [sample.reshape(1, -1, 1, 1) for sample in samples]

        samples_: list[D.DistributionDraw] = [self.transform_noise(z_) for
                                              z_ in samples]

        return samples_, samples

    @final
    @override
    def create_distribution(
        self,
        input_: torch.Tensor
    ) -> D.DistributionDraw:
        return D.DistributionDraw.from_grayscale_weights(input_.squeeze())

    @override
    def _additional_repr_(self, sep: str) -> str:
        to_return = ""

        if hasattr(self, "data_"):
            to_return += f"n_data={len(self.data_):_}" + sep

        if self.samples_cache:
            to_return += f"n_cached_samples={self.n_cached_samples:_}" + sep

        if self.n_steps:
            to_return += f"len_chain={len(self.chains[0]):_}" + sep
            to_return += f"n_steps={self.n_steps:_}" + sep

        return to_return

    # noinspection DuplicatedCode,PyTypeChecker
    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)
        generator = copy.copy(self.generator_)
        transform_out = copy.copy(self.transform_out_)
        noise_sampler = copy.copy(self.noise_sampler_)
        data = copy.copy(self.data_)

        new = self.__class__(
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.copy(self.chains)
        new.samples_cache = copy.copy(self.samples_cache)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    # noinspection DuplicatedCode,PyTypeChecker
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        # Manually create a new dictionary and copy the items
        hamiltorch_kwargs = {k: copy.deepcopy(v, memo) for k, v in
                             self._hamiltorch_kwargs.items()}

        generator = copy.deepcopy(self.generator_, memo)
        transform_out = copy.deepcopy(self.transform_out_, memo)
        noise_sampler = copy.deepcopy(self.noise_sampler_, memo)
        data = copy.deepcopy(self.data_, memo)

        new = self.__class__(
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.deepcopy(self.chains, memo)
        new.samples_cache = copy.deepcopy(self.samples_cache, memo)
        new.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new

    @override
    def _getstate_half_(self, state):
        state = super()._getstate_half_(state)
        state["chains"] = self._set_half_dtype(self.chains)
        state["samples_cache"] = self._set_half_dtype(self.samples_cache)
        return state

    @classmethod
    @override
    def _load_half_(
        cls,
        new: Self,
        device_: torch.device,
        dtype_: torch.dtype
    ) -> None:
        super()._load_half_(new, device_, dtype_)
        new.chains = new._set_normal_dtype(new.chains)
        new.samples_cache = new._set_normal_dtype(new.samples_cache)


class LatentMCMCPosteriorSampler(BaseLatentMCMCPosteriorSampler):
    def __init__(
        self,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        sampler: Sampler = Sampler.HMC,
        n_walkers: int = 1,
        n_workers: int = None,
        parallel: bool = False,
        log_likelihood_fn=log_likelihood_latent,
        log_prior_fn=log_prior,
        save_samples: bool = True,
        use_half: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            num_steps_per_sample=num_steps_per_sample,
            burn=burn,
            step_size=step_size,
            sampler=sampler,
            save_samples=save_samples,
            log_likelihood_fn=log_likelihood_fn,
            log_prior_fn=log_prior_fn,
            use_half=use_half,
            **kwargs
        )
        self.n_workers = n_workers
        self.n_walkers = n_walkers
        self.parallel = parallel
        self.chains: list[list[torch.Tensor]] = self._create_empty_chains(
            n_walkers)

    @override
    @timeit_to_total_time
    def run(
        self,
        n_steps: int = None,
        burn: int = None,
        parallel: bool = None,
        n_workers: int = None,
        n_walkers: int = None,
        seeds: Sequence[SeedT] = None,
        **kwargs,
    ) -> Self:
        check_is_fitted(self,
                        ["generator_", "transform_out_",
                         "noise_sampler_", "data_"])

        n_workers = n_workers if n_workers is not None else self.n_workers
        n_walkers = n_walkers if n_walkers is not None else self.n_walkers
        parallel = parallel if parallel is not None else self.parallel

        seeds = seeds if seeds is not None else torch.arange(n_walkers)
        if len(seeds) != n_walkers:
            msg = "The number of seeds must match the number of walkers."
            _log.error(msg)
            raise ValueError(msg)

        def _prior():
            return self.noise_sampler_(1).squeeze()

        hamiltorch_kwargs = self._get_hamiltorch_kwargs(num_samples=n_steps,
                                                        burn=burn, **kwargs)
        debug = hamiltorch_kwargs["debug"]

        chain = hamiltorch.util.setup_chain(hamiltorch.sample, _prior,
                                            hamiltorch_kwargs)

        params_hmc = hamiltorch.util.multi_chain(chain, n_workers, seeds,
                                                 parallel)

        log_list = []
        for i, params in enumerate(params_hmc):
            if debug == 2:
                params, log = params
                log_list.append(log)
            params = self._set_half_dtype(params)

            self.chains[i].extend(params)
            self.n_steps += len(params)

        self.log = log_list

        return self

    @override
    def get_chain(
        self,
        flat: bool = False,
        thin: int = 1,
        discard: int = 0
    ) -> torch.Tensor:
        samples_per_chain = [torch.stack(chain) for chain in self.chains]
        to_return = torch.stack(samples_per_chain, dim=1)
        to_return = to_return[discard::thin]

        if flat:
            to_return = to_return.flatten(0, 1)
        to_return = self._set_normal_dtype(to_return)

        return to_return

    @override
    def _additional_repr_(self, sep: str) -> str:
        to_return = super()._additional_repr_(sep)

        if self.n_workers is not None:
            to_return += f"n_workers={self.n_workers}" + sep

        if self.n_walkers > 1:
            to_return += f"n_walkers={self.n_walkers}" + sep

        if self.parallel:
            to_return += "parallel=True" + sep

        return to_return

    # noinspection DuplicatedCode,PyTypeChecker
    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)
        generator = copy.copy(self.generator_)
        transform_out = copy.copy(self.transform_out_)
        noise_sampler = copy.copy(self.noise_sampler_)
        data = copy.copy(self.data_)

        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.copy(self.chains)
        new.samples_cache = copy.copy(self.samples_cache)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    # noinspection DuplicatedCode,PyTypeChecker
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        # Manually create a new dictionary and copy the items
        hamiltorch_kwargs = {k: copy.deepcopy(v, memo) for k, v in
                             self._hamiltorch_kwargs.items()}

        generator = copy.deepcopy(self.generator_, memo)
        transform_out = copy.deepcopy(self.transform_out_, memo)
        noise_sampler = copy.deepcopy(self.noise_sampler_, memo)
        data = copy.deepcopy(self.data_, memo)

        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            sampler=hamiltorch_kwargs.pop("sampler"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.deepcopy(self.chains, memo)
        new.samples_cache = copy.deepcopy(self.samples_cache, memo)
        new.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


class NUTSPosteriorSampler(LatentMCMCPosteriorSampler):
    def __init__(
        self,
        num_samples: int = 10,
        num_steps_per_sample: int = 10,
        burn: int = 10,
        step_size: float = 0.1,
        desired_accept_rate: float = 0.6,
        n_walkers: int = 1,
        n_workers: int = None,
        parallel: bool = False,
        log_likelihood_fn=log_likelihood_latent,
        log_prior_fn=log_prior,
        save_samples: bool = True,
        use_half: bool = False,
        **kwargs,
    ):
        super().__init__(
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
            log_likelihood_fn=log_likelihood_fn,
            log_prior_fn=log_prior_fn,
            use_half=use_half,
            **kwargs
        )

    # noinspection DuplicatedCode,PyTypeChecker
    def __copy__(self):
        hamiltorch_kwargs = copy.copy(self._hamiltorch_kwargs)
        hamiltorch_kwargs.pop(
            "sampler")  # The sampler is already set in the constructor
        generator = copy.copy(self.generator_)
        transform_out = copy.copy(self.transform_out_)
        noise_sampler = copy.copy(self.noise_sampler_)
        data = copy.copy(self.data_)

        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            desired_accept_rate=hamiltorch_kwargs.pop("desired_accept_rate"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.copy(self.chains)
        new.samples_cache = copy.copy(self.samples_cache)
        new.log = copy.copy(self.log)

        new.__dict__.update(self.__dict__)

        return new

    # noinspection DuplicatedCode,PyTypeChecker
    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}

        # Manually create a new dictionary and copy the items
        hamiltorch_kwargs = {k: copy.deepcopy(v, memo) for k, v in
                             self._hamiltorch_kwargs.items()}

        hamiltorch_kwargs.pop(
            "sampler")  # The sampler is already set in the constructor
        generator = copy.deepcopy(self.generator_, memo)
        transform_out = copy.deepcopy(self.transform_out_, memo)
        noise_sampler = copy.deepcopy(self.noise_sampler_, memo)
        data = copy.deepcopy(self.data_, memo)

        new = self.__class__(
            n_workers=self.n_workers,
            n_walkers=self.n_walkers,
            log_likelihood_fn=self.log_likelihood_fn,
            log_prior_fn=self.log_prior_fn,
            num_samples=hamiltorch_kwargs.pop("num_samples"),
            num_steps_per_sample=hamiltorch_kwargs.pop("num_steps_per_sample"),
            burn=hamiltorch_kwargs.pop("burn"),
            step_size=hamiltorch_kwargs.pop("step_size"),
            desired_accept_rate=hamiltorch_kwargs.pop("desired_accept_rate"),
            parallel=self.parallel,
            save_samples=self.save_samples,
            use_half=self.use_half,
            **hamiltorch_kwargs
        ).fit(
            generator=generator, transform_out=transform_out,
            noise_sampler=noise_sampler, data=data
        )

        new.chains = copy.deepcopy(self.chains, memo)
        new.samples_cache = copy.deepcopy(self.samples_cache, memo)
        new.log = copy.deepcopy(self.log, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


class ExplicitPosteriorPiN(ExplicitPosteriorSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Raise a warning of deprecation
        warnings.warn(
            "ExplicitPosteriorPiN is deprecated. Use "
            "ExplicitPosteriorSampler instead.",
            DeprecationWarning, stacklevel=2,
        )


# noinspection DuplicatedCode
def __main():
    from icecream import ic
    from pathlib import Path
    logging.set_level(logging.DEBUG)

    def _noise_sampler(size):
        """Dummy noise_sampler."""
        return torch.rand((size, 128, 1, 1)).to(conf.device)

    def _generator(z):
        """Dummy generator."""
        return torch.rand((1, 32, 32)).to(conf.device)

    def _transform_out(x):
        """Dummy transform_out."""
        return x

    BASE_LATENT_MCMC_POSTERIOR_SAMPLER = False
    LATENT_MCMC_POSTERIOR_SAMPLER = False
    NUTS_POSTERIOR_SAMPLER = True

    input_size = 128
    output_size = (32, 32)
    n_data = 100
    data_ = torch.randint(0, output_size[0] * output_size[1], (n_data,)).to(
        conf.device)

    # ======== BaseLatentMCMCPosteriorSampler ========
    if BASE_LATENT_MCMC_POSTERIOR_SAMPLER:
        # Test representation
        ic()
        posterior = BaseLatentMCMCPosteriorSampler()
        ic(posterior)

        # Test fit
        try:
            posterior.draw()
        except Exception as e:
            ic(e)
        try:
            posterior.sample(10)
        except Exception as e:
            ic(e)
        posterior.fit(_generator, _transform_out, _noise_sampler, data_)
        ic("fit test", posterior)

        # Test draw and rvs
        posterior.draw()
        posterior.sample(10)
        ic("sample test", posterior)

        # Test copy and deep copy
        posterior_ = copy.copy(posterior)
        ic("copy test", posterior_)
        ic(posterior_._hamiltorch_kwargs)
        posterior_ = copy.deepcopy(posterior)
        ic("deep copy test", posterior_)
        ic(posterior_._hamiltorch_kwargs)

    # ======== LatentMCMCPosteriorSampler ========
    if LATENT_MCMC_POSTERIOR_SAMPLER:
        ic()
        # Test representation
        posterior = LatentMCMCPosteriorSampler()
        ic(posterior)
        ic(LatentMCMCPosteriorSampler(parallel=True))
        ic(LatentMCMCPosteriorSampler(n_walkers=2))
        ic(LatentMCMCPosteriorSampler(n_workers=2, parallel=True))

        # Test fit
        try:
            posterior.draw()
        except Exception as e:
            ic(e)
        try:
            posterior.sample(10)
        except Exception as e:
            ic(e)
        posterior.fit(_generator, _transform_out, _noise_sampler, data_)
        ic("fit test", posterior)

        # Test draw and rvs
        posterior.draw()
        posterior.sample(10)
        ic("sample test", posterior)

        # Test copy and deep copy
        posterior_ = copy.copy(posterior)
        ic("copy test", posterior_)
        posterior_ = copy.deepcopy(posterior)
        ic("deep copy test", posterior_)

    # ======== NUTSPosteriorSampler ========
    if NUTS_POSTERIOR_SAMPLER:
        ic()
        # Test representation
        posterior = NUTSPosteriorSampler(use_half=True)
        ic(posterior)
        ic(NUTSPosteriorSampler(parallel=True))
        ic(NUTSPosteriorSampler(n_walkers=2))
        ic(NUTSPosteriorSampler(n_workers=2, parallel=True))

        # Test fit
        try:
            posterior.draw()
        except Exception as e:
            ic(e)
        try:
            posterior.sample(10)
        except Exception as e:
            ic(e)
        posterior.fit(_generator, _transform_out, _noise_sampler, data_)
        ic("fit test", posterior)

        # Test draw and rvs
        posterior.draw()
        posterior.sample(10)
        ic("sample test", posterior)

        # Test copy and deep copy
        posterior_ = copy.copy(posterior)
        ic("copy test", posterior_)
        ic(posterior_._hamiltorch_kwargs)
        posterior_ = copy.deepcopy(posterior)
        ic("deep copy test", posterior_)
        ic(posterior_._hamiltorch_kwargs)

        with logging.register_total_time(_log) as timer:
            posterior.sample(50)

        # Save test
        with logging.register_total_time(_log) as timer:
            SAVE_PATH = Path("posterior_sampler.pkl")
            ic(posterior.noise_sampler_)
            posterior.save(SAVE_PATH)
            posterior_ = NUTSPosteriorSampler.load(SAVE_PATH)
            ic("save test", posterior_)
            ic(SAVE_PATH.stat().st_size)
            SAVE_PATH.unlink()

        # Save test with gzip
        with logging.register_total_time(_log) as timer:
            SAVE_PATH = Path("posterior_sampler.pkl.gz")
            posterior.save(SAVE_PATH)
            posterior_ = NUTSPosteriorSampler.load(SAVE_PATH)
            ic("save test with gzip", posterior_)
            ic(SAVE_PATH.stat().st_size)
            SAVE_PATH.unlink()

        # Save test with gzip and without half precision
        NUTSPosteriorSampler.SAVE_HALF_PRECISION = False
        with logging.register_total_time(_log) as timer:
            SAVE_PATH = Path("posterior_sampler.pkl.gz")
            posterior.save(SAVE_PATH)
            posterior_ = NUTSPosteriorSampler.load(SAVE_PATH)
            ic("save test with gzip and without half precision", posterior_)
            ic(SAVE_PATH.stat().st_size)
            SAVE_PATH.unlink()


if __name__ == '__main__':
    __main()
