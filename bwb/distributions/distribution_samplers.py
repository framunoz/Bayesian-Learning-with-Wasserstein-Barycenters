"""
This module contains the classes that represent the distribution
samplers. These classes are used to sample distributions from a set of
models, and they are divided into two main categories:
discrete and continuous samplers.
"""

import abc
import collections as c
import copy
import gzip
import pickle
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Protocol,
    Self,
    final,
    override,
)

import torch

import bwb.distributions as D
import bwb.logging_ as logging
from bwb.config import config
from bwb.distributions.models import DiscreteModelsSetP
from bwb.protocols import HasDeviceDType
from bwb.utils import (
    PathT,
    SeedT,
    check_is_fitted,
    set_generator,
    timeit_to_total_time,
)

_log = logging.get_logger(__name__)

__all__ = [
    "BaseGeneratorDistribSampler",
    "ContinuousDistribSampler",
    "DiscreteDistribSampler",
    "DistributionSampler",
    "GeneratorDistribSampler",
    "GeneratorP",
    "UniformDiscreteSampler",
]


class GeneratorP(Protocol):
    """
    Protocol for a generator model.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class DistributionSampler[DistributionT](HasDeviceDType, metaclass=abc.ABCMeta):
    r"""
    Base class for distributions that sampling other distributions.
    i.e. it represents a distribution :math:`\Lambda(dm) \in \mathcal{P}(
    \mathcal{M)})`, where :math:`\mathcal{M}` is the set of models.
    """

    def __init__(self) -> None:
        self.total_time = 0.0  # Total time to draw samples

    @override
    @property
    def device(self) -> torch.device:
        return self.sample(1)[0].device

    @override
    @property
    def dtype(self) -> torch.dtype:
        return self.sample(1)[0].dtype

    @abc.abstractmethod
    def draw(self, seed: SeedT = None) -> DistributionT:
        """
        Draw a sample.

        :param seed: The seed to draw the sample.
        :return: The drawn sample.
        """
        ...

    @abc.abstractmethod
    def sample(self, size: int = 1, seed: SeedT = None) -> list[DistributionT]:
        """
        Samples as many distributions as the ``size`` parameter
        indicates.

        :param size: The number of samples to draw.
        :param seed: Seed for the random number generator. If None, a
            random seed will be used.
        :return: A sequence of sampled distributions.
        """
        ...

    def save(self, filename: PathT) -> None:
        """
        Save the object to a file.

        :param filename: The path to the file.
        :return: None
        """
        filename: Path = Path(filename)

        if filename.suffix not in [".pkl", ".gz"]:
            msg = "The file must have the extension '.pkl' or '.gz'."
            logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)

        if filename.suffix == ".gz":
            f = gzip.open(filename, "wb")
        else:
            f = open(filename, "wb")

        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    @classmethod
    def load(cls, filename: PathT) -> Self:
        """
        Load the object from a file.

        :param filename: The path to the file.
        :return: The loaded object.
        """
        filename: Path = Path(filename)

        if filename.suffix == ".gz":
            f = gzip.open(filename, "rb")
        else:
            f = open(filename, "rb")

        obj = pickle.load(f)
        f.close()

        return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# noinspection PyAttributeOutsideInit
class DiscreteDistribSampler[DistributionT](DistributionSampler[DistributionT]):
    r"""
    Base class for distributions that have a discrete set of models.
    i.e. where the set of models is :math:`|\mathcal{M}| < +\infty`.

    As the support is discrete, the distribution can be represented as
    a vector of probabilities, and therefore, the sampling process is
    reduced to drawing an index from a multinomial distribution. This
    property allows to save the samples and the number of times each
    model has been sampled, to get statistics about the sampling process.
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

    def fit(self, models: DiscreteModelsSetP[DistributionT], **kwargs) -> Self:
        """Fit the distribution."""
        assert isinstance(models, DiscreteModelsSetP), (
            "The models must be a DiscreteModelsSet.\n"
            f"Missing methods: {(set(dir(DiscreteModelsSetP))
                                 - set(dir(models))
                                 - {'_abc_impl',
                                    '_is_runtime_protocol',
                                    '__abstractmethods__'})}."
        )

        # The set of models
        self.models_: DiscreteModelsSetP[DistributionT] = models

        self.models_index_: torch.Tensor = torch.arange(
            len(models), device=self.device
        )  # The index of the models

        # The probabilities need to be set!
        return self

    @override
    @property
    def device(self) -> torch.device:
        return self.models_.get(0).device

    @override
    @property
    def dtype(self) -> torch.dtype:
        return self.models_.get(0).dtype

    def get_model(self, i: int) -> DistributionT:
        """Get the model with index i."""
        check_is_fitted(self, ["models_"])
        if self._models_cache.get(i) is None:
            self._models_cache[i] = self.models_.get(i)
        return self._models_cache[i]

    def _draw(self, seed: SeedT = None) -> tuple[DistributionT, int]:
        """To use template pattern on the draw method."""
        rng: torch.Generator = set_generator(seed=seed, device=self.device)

        i = torch.multinomial(
            input=self.probabilities_, num_samples=1, generator=rng
        ).item()
        i = int(i)

        return self.get_model(i), i

    @timeit_to_total_time
    @override
    def draw(self, seed: SeedT = None) -> DistributionT:
        """Draw a sample."""
        check_is_fitted(self, ["models_", "probabilities_"])
        to_return, i = self._draw(seed)
        if self.save_samples:  # Register the sample
            self.samples_history.append(i)
            self.samples_counter[i] += 1
        return to_return

    def _sample(
        self, size: int = 1, seed: SeedT = None
    ) -> tuple[list[DistributionT], list[int]]:
        """
        Samples as many distributions as the ``size`` parameter
        indicates.
        """
        rng: torch.Generator = set_generator(seed=seed, device=self.device)

        indices = torch.multinomial(
            input=self.probabilities_,
            num_samples=size,
            replacement=True,
            generator=rng,
        )
        indices = indices.tolist()
        return [self.get_model(i) for i in indices], indices

    @timeit_to_total_time
    @override
    def sample(self, size: int = 1, seed: SeedT = None) -> list[DistributionT]:
        """
        Samples as many distributions as the ``size`` parameter
        indicates.
        """
        check_is_fitted(self, ["models_", "probabilities_"])
        to_return, list_indices = self._sample(size, seed)
        if self.save_samples:  # Register the samples
            self.samples_history.extend(list_indices)
            self.samples_counter.update(list_indices)
        return to_return

    @override
    def __repr__(self) -> str:
        to_return = self.__class__.__name__ + "("

        if self.save_samples:
            to_return += f"samples={len(self.samples_history)}, "

        if to_return[-2:] == ", ":  # Remove the last comma
            to_return = to_return[:-2]

        to_return += ")"

        return to_return


# noinspection PyAttributeOutsideInit
class UniformDiscreteSampler[DistributionT](
    DiscreteDistribSampler[DistributionT]
):
    r"""
    A class representing a distribution sampler with a discrete set of
    models, and the probabilities are set to be uniform.

    This class inherits from the `DiscreteDistribSampler` class and
    provides methods to fit the sampler to a set of discrete models and
    generate samples from the fitted sampler.

    Attributes:
        probabilities_: A torch.Tensor representing the probabilities
        of each model in the sampler.
        support_: The indices of the models in the sampler.

    Methods:
        fit: Fits the sampler to a set of discrete models.
        __repr__: Returns a string representation of the sampler.

    """

    @timeit_to_total_time
    @override
    def fit(self, models: DiscreteModelsSetP[DistributionT], **kwargs) -> Self:
        super().fit(models)
        self.probabilities_: torch.Tensor = torch.ones(
            len(models),
            device=self.device,
            dtype=self.dtype,
        ) / len(models)

        self.support_ = self.models_index_

        self._fitted = True

        return self

    @override
    def __repr__(self) -> str:
        to_return = self.__class__.__name__

        if not self._fitted:
            to_return += "()"
            return to_return

        to_return += "("
        to_return += f"n_models={len(self.models_)}, "
        if self.save_samples:
            to_return += f"samples={len(self.samples_history)}, "

        if to_return[-2:] == ", ":  # Remove the last comma
            to_return = to_return[:-2]

        to_return += ")"

        return to_return


class ContinuousDistribSampler[DistributionT](
    DistributionSampler[DistributionT], abc.ABC
):
    r"""
    Class for distributions that have a continuous set of models. i.e.
    where the set of models is :math:`|\mathcal{M}| = +\infty`.
    """

    ...


# noinspection PyAttributeOutsideInit
class BaseGeneratorDistribSampler[DistributionT](
    ContinuousDistribSampler[DistributionT], metaclass=abc.ABCMeta
):
    """
    Base class for distributions that have a continuous set of models,
    and the models can be generated by a generator.
    """

    SAVE_SAMPLES: bool = False
    SAVE_HALF_PRECISION: bool = True
    noise_sampler_: Callable[[int, torch.Generator], torch.Tensor]
    generator_: GeneratorP
    transform_out_: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self, save_samples: Optional[bool] = None, use_half: bool = False
    ) -> None:
        super().__init__()
        self.save_samples: bool = save_samples or self.SAVE_SAMPLES
        self.use_half: bool = use_half
        self.samples_history: list[torch.Tensor] = []
        self._fitted: bool = False

    def _set_half_dtype[T](self, value: T, force: bool = False) -> T:
        """
        Set the half precision dtype to the tensor, if the attribute
        ``use_half`` is True.

        :param value: The value to set the half precision dtype.
        :param force: If True, the half precision dtype will be set
            regardless of the attribute ``use_half``. Default is False.
        :return: The value with the half precision dtype.
        """
        if not self.use_half and not force:
            return value

        if isinstance(value, torch.Tensor):
            return value.half()

        if not isinstance(value, list):
            msg = "The value must be a tensor or a list of tensors."
            logging.raise_error(msg, _log, TypeError)

        return [self._set_half_dtype(v, force) for v in value]

    def _set_normal_dtype[T](self, value: T, force: bool = False) -> T:
        """
        Set the normal precision dtype to the tensor, if the attribute
        ``use_half`` is True.

        :param value: The value to set the normal precision dtype.
        :param force: If True, the normal precision dtype will be set
            regardless of the attribute ``use_half``. Default is False.
        :return: The value with the normal precision dtype.
        """
        if not self.use_half and not force:
            return value

        if isinstance(value, torch.Tensor):
            return value.to(dtype=self.dtype)

        if not isinstance(value, list):
            msg = "The value must be a tensor or a list of tensors."
            logging.raise_error(msg, _log, TypeError)

        return [self._set_normal_dtype(v, force) for v in value]

    @classmethod
    def set_save_samples(cls, save_samples: bool) -> None:
        """Set the default attribute ``save_samples`` to the class."""
        if not isinstance(save_samples, bool):
            raise TypeError("The save_samples must be a boolean.")
        cls.SAVE_SAMPLES = save_samples

    @abc.abstractmethod
    def create_distribution(self, input_: torch.Tensor) -> DistributionT:
        """
        Creates a distribution from the input tensor.

        :param input_: The input tensor.
        :return: The distribution.
        """
        ...

    @abc.abstractmethod
    def _draw(self, seed: SeedT = None) -> tuple[DistributionT, torch.Tensor]:
        """
        Draw a sample and return the distribution and the noise. This
        method is used to implement the draw method.
        """
        pass

    @abc.abstractmethod
    def _sample(
        self, size: int = 1, seed: SeedT = None
    ) -> tuple[list[DistributionT], list[torch.Tensor]]:
        """
        Samples as many distributions as the `size` parameter indicates.
        """
        pass

    def fit(
        self,
        generator: GeneratorP,
        transform_out: Callable[[torch.Tensor], torch.Tensor],
        noise_sampler: Callable[[int], torch.Tensor],
        **kwargs,
    ) -> Self:
        """
        Fits the distribution sampler.

        :param generator: The generator model.
        :type generator: GeneratorP
        :param transform_out: A callable function that transforms the
            output of the generator.
        :type transform_out: Callable[[torch.Tensor], torch.Tensor]
        :param noise_sampler: A callable function that generates noise
            samples.
        :type noise_sampler: Callable[[int], torch.Tensor]
        :raises ValueError: If the noise sampler does not return a tensor.
        :raises ValueError: If the generator fails to generate a sample
            from the noise.
        :raises ValueError: If the transform_out fails to transform the
            output of the generator.
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
                    "The noise sampler must return a tensor, "
                    f"but it returns a {type(noise)}"
                )
            if noise.shape[0] != 42:

                raise ValueError(
                    "The noise sampler must return a tensor of shape "
                    "(42, ...), but it returns a tensor of shape "
                    f"{noise.shape}"
                )
        except Exception as _:
            raise ValueError("The noise sampler must accept an integer.")
        self.noise_sampler_: Callable[[int], torch.Tensor] = noise_sampler

        # Validation of the generator
        try:
            output = generator(noise)
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    "The generator must return a tensor, but it returns a "
                    f"{type(output)}"
                )
            if output.device != self.device:
                raise ValueError(
                    "The generator must return a tensor with the same "
                    "device as the noise."
                )
        except Exception as e:
            _log.error(e)
            raise ValueError(
                "The generator must be able to generate a sample from "
                "the noise."
            )
        self.generator_ = generator

        # Validation of the transform_out
        try:
            noise = noise_sampler(1)
            result = generator(noise)
            output = transform_out(result)
            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    "The transform_out must return a tensor, "
                    f"but it returns a {type(output)}"
                )
            if output.device != self.device:
                raise ValueError(
                    "The transform_out must return a tensor with the "
                    "same device as the noise."
                )
        except Exception as e:
            _log.error(e)
            raise ValueError(
                "The transform_out must be able to transform the output "
                "of the generator."
            )
        # Validation that the output of the transform_out can be a distribution
        try:
            _ = self.create_distribution(output)
        except Exception as e:
            _log.error(e)
            raise ValueError(
                "The transform_out must return a tensor that can be "
                "transformed into a distribution."
            )
        self.transform_out_ = transform_out

        self._fitted = True
        return self

    @override
    @property
    def dtype(self) -> torch.dtype:
        return self.noise_sampler_(1).dtype

    @override
    @property
    def device(self) -> torch.device:
        return self.noise_sampler_(1).device

    def _additional_repr_(self, sep: str) -> str:
        """
        Override this method to add additional information to the
        __repr__ method.

        :return: The additional information.
        """
        to_return = ""

        if self.save_samples:
            to_return += f"samples={len(self.samples_history)}" + sep

        return to_return

    @override
    def __repr__(self, sep=", ") -> str:
        to_return = self.__class__.__name__ + "("

        to_return += self._additional_repr_(sep)

        if to_return[-len(sep) :] == sep:
            to_return = to_return[: -len(sep)]

        to_return += ")"

        return to_return

    @final
    def transform_noise(self, noise: torch.Tensor) -> DistributionT:
        """
        Transforms the given noise tensor using the generator network.

        :param noise: The noise tensor.
        :type noise: torch.Tensor
        :raises NotFittedError: If the distribution sampler is not fitted.
        :return: The transformed noise tensor.
        :rtype: DistributionT
        """
        # Set the normal precision dtype
        noise = self._set_normal_dtype(noise)

        with torch.no_grad():
            gen_output = self.generator_(noise)

        output = self.transform_out_(gen_output)

        return self.create_distribution(output)

    @final
    @override
    def draw(self, seed: SeedT = None) -> DistributionT:
        # Check if the distribution sampler is fitted
        if not self._fitted:
            attributes = ["generator_", "transform_out_", "noise_sampler_"]
            check_is_fitted(self, attributes)

        to_return, noise = self._draw(seed)
        if self.save_samples:
            noise = self._set_half_dtype(noise)
            self.samples_history.append(noise)

        return to_return

    @final
    @override
    def sample(self, size: int = 1, seed: SeedT = None) -> list[DistributionT]:
        # Check if the distribution sampler is fitted
        if not self._fitted:
            attributes = ["generator_", "transform_out_", "noise_sampler_"]
            check_is_fitted(self, attributes)

        to_return, noises = self._sample(size, seed)
        if self.save_samples:
            noises = self._set_half_dtype(noises)
            self.samples_history.extend(noises)

        return to_return

    def _get_noise(self, size: int = 1, seed: SeedT = None) -> torch.Tensor:
        """
        Get noise samples. If the seed is None, a random seed will be
        used. If the noise sampler fails to generate noise with the seed,
        a warning will be raised and the noise will be
        generated without the seed.

        :param size: The number of noise samples to generate.
        :param seed: The seed to generate the noise samples.
        :return: The noise samples.
        """
        # If the seed is None, generate the noise without the seed
        if seed is None:
            return self.noise_sampler_(size)

        # Otherwise, try to generate the noise with the seed
        _ = set_generator(seed=seed, device=self.device)
        try:
            # noinspection PyArgumentList
            noise = self.noise_sampler_(size, seed=seed)
        except KeyError as e:
            noise = self.noise_sampler_(size)
            msg = (
                "Failed to generate noise with seed. The noise "
                f"will be generated without the seed. {e}"
            )
            logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)
        except Exception as e:
            noise = self.noise_sampler_(size)
            msg = (
                "Failed to generate noise with seed. The noise "
                f"will be generated without the seed. {e}"
            )
            logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)
        return noise

    def __copy__(self) -> Self:
        generator = copy.copy(self.generator_)
        transform_out = copy.copy(self.transform_out_)
        noise_sampler = copy.copy(self.noise_sampler_)

        new = self.__class__(save_samples=self.save_samples).fit(
            generator, transform_out, noise_sampler
        )
        new.samples_history = copy.copy(self.samples_history)

        new.__dict__.update(self.__dict__)

        return new

    def __deepcopy__(self, memo=None) -> Self:
        if memo is None:
            memo = {}

        generator = copy.deepcopy(self.generator_, memo)
        transform_out = copy.deepcopy(self.transform_out_, memo)
        noise_sampler = copy.deepcopy(self.noise_sampler_, memo)

        new = self.__class__(save_samples=self.save_samples).fit(
            generator, transform_out, noise_sampler
        )
        new.samples_history = copy.deepcopy(self.samples_history, memo)

        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new

    @override
    def __getstate__(self) -> dict:
        msg = (
            "The generator_, transform_out_ and noise_sampler_ "
            "attributes will not be saved. Use fit method to fit "
            "these attributes."
        )
        logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)

        # Remove the generator, transform_out and noise_sampler attributes
        state = copy.copy(self.__dict__)
        state.pop("generator_", None)
        state.pop("transform_out_", None)
        state.pop("noise_sampler_", None)

        if self.SAVE_HALF_PRECISION and not self.use_half:
            msg = (
                "The samples will be saved in half precision. To "
                "disable this behavior, set the "
                "attribute SAVE_HALF_PRECISION to False."
            )
            logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)
            state = self._getstate_half_(state)

        return state

    def _getstate_half_(self, state) -> dict:
        """
        To use template pattern on the __getstate__ method.

        :param state: The state to return.
        :return: The state.
        """
        state["samples_history"] = self._set_half_dtype(
            self.samples_history, force=True
        )
        return state

    @classmethod
    @override
    def load(cls, filename: PathT) -> Self:
        new = super().load(filename)
        new._fitted = False

        if cls.SAVE_HALF_PRECISION and not new.use_half:
            if not (hasattr(new, "device") and hasattr(new, "dtype")):
                msg = (
                    "The device and dtype were not saved. The samples "
                    "will be loaded with the default device and dtype."
                )
                logging.raise_warning(msg, _log, RuntimeWarning, stacklevel=2)
                cls._load_half_(new, config.device, config.dtype)
            else:
                cls._load_half_(new, new.device, new.dtype)

        return new

    @classmethod
    def _load_half_(cls, new: Self, device_: device, dtype_: dtype) -> None:
        """
        To use template pattern on the load method.

        :param new: The new instance.
        :param device_: The device to load the samples.
        :param dtype_: The dtype to load the samples.
        :return: The new instance.
        """
        new.samples_history = new._set_normal_dtype(
            new.samples_history, force=True
        )


# noinspection PyAttributeOutsideInit
class GeneratorDistribSampler(BaseGeneratorDistribSampler[D.DistributionDraw]):
    r"""
    Class for distributions that have a continuous set of models,
    and the models can be generated by a generator model.
    """

    @final
    @override
    def create_distribution(self, input_: torch.Tensor) -> D.DistributionDraw:
        return D.DistributionDraw.from_grayscale_weights(input_.squeeze())

    @final
    @override
    def _draw(
        self, seed: SeedT = None
    ) -> tuple[D.DistributionDraw, torch.Tensor]:
        noise: torch.Tensor = self._get_noise(1, seed)
        to_return: D.DistributionDraw = self.transform_noise(noise)
        return to_return, noise

    @final
    @override
    def _sample(
        self, size: int = 1, seed: SeedT = None
    ) -> tuple[list[D.DistributionDraw], list[torch.Tensor]]:
        noises = [noise for noise in self._get_noise(size, seed)]
        to_return: list[D.DistributionDraw] = [
            self.transform_noise(noise.unsqueeze(0)) for noise in noises
        ]
        return to_return, noises


def __main() -> None:
    """
    Main function for testing purposes.
    """
    from pathlib import Path

    from icecream import ic

    input_size = 128
    output_size = (32, 32)

    def noise_sampler(size) -> torch.Tensor:
        """Dummy noise_sampler."""
        return torch.rand((size, input_size, 1, 1)).to(config.device)

    # noinspection PyUnusedLocal
    def generator(z) -> torch.Tensor:
        """Dummy generator."""
        return torch.rand((1, output_size[0], output_size[1])).to(config.device)

    def transform_out(x) -> torch.Tensor:
        """Dummy transform_out."""
        return x

    # Test that GeneratorDistribSampler can be copied
    distr_sampler = GeneratorDistribSampler(save_samples=True)
    ic(distr_sampler)

    try:
        distr_sampler.sample(1)
    except Exception as e:
        ic(e)

    try:
        distr_sampler.sample(10)
    except Exception as e:
        ic(e)

    distr_sampler.fit(generator, transform_out, noise_sampler)
    # distr_sampler.draw()
    distr_sampler.sample(10)
    ic(distr_sampler)

    distr_sampler_ = copy.copy(distr_sampler)
    ic(distr_sampler_)

    distr_sampler_ = copy.deepcopy(distr_sampler)
    ic(distr_sampler_)

    # Save and load test
    filename = Path("test_half.pkl")
    distr_sampler.sample(1_000)
    distr_sampler.save(filename)
    distr_sampler_ = GeneratorDistribSampler.load(filename)
    ic(distr_sampler_)
    ic(filename.stat().st_size)
    # Delete file
    filename.unlink()

    # Save and load test with half precision
    GeneratorDistribSampler.SAVE_HALF_PRECISION = False
    filename = Path("test.pkl")
    ic(distr_sampler.samples_history[0].dtype)
    ic(distr_sampler.SAVE_HALF_PRECISION)
    distr_sampler.save(filename)
    distr_sampler_ = GeneratorDistribSampler.load(filename)
    ic(distr_sampler_.samples_history[0].dtype)
    ic(distr_sampler_.SAVE_HALF_PRECISION)
    ic(distr_sampler_)
    ic(filename.stat().st_size)
    # Delete file
    filename.unlink()


if __name__ == "__main__":
    __main()
