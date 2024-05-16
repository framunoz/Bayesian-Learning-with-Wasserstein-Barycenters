"""Configuration module for the library. This module contains the
configuration class and some helper functions to change the
configuration. The configuration is a singleton, so it is shared across
the whole library."""
import threading
import typing as t
import warnings

import torch

from bwb.logging_ import log_config

_log = log_config.get_logger(__name__)


# noinspection DuplicatedCode
class _SingletonMeta(type):
    """Metaclass to implements Singleton Pattern. Obtained from
    https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class CudaNotAvailableError(Exception):
    """
    Exception raised when CUDA is not available.
    """

    def __init__(self, message: str, n_cuda: int = 0):
        super().__init__(message)
        self.n_cuda = n_cuda


# IDEA: Make the config an observer, so that when the dtype or device
# is changed, all the tensors are changed. This can be done by adding a
# list of observers and calling a method when the configuration is changed.
class Config(metaclass=_SingletonMeta):
    """
    Configuration class for the library. This class is a singleton, so
    it is shared across the whole library.
    """
    # The dtype selected by default
    dtype = torch.float32

    # Use cuda if is possible
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    # Settings for the smallest number
    _eps = None

    @property
    def eps(self) -> torch.Tensor:
        """
        Get the positive minimum for parameters. This is usually
        slightly larger than zero to avoid numerical errors.

        :return: The positive minimum for parameters.
        """
        if self._eps is None:
            self._eps = torch.tensor(torch.finfo(self.dtype).eps)
        return self._eps

    @eps.setter
    def eps(self, value):
        self._eps = torch.tensor(value)

    @classmethod
    def use_half_precision(cls):
        """
        Use half precision (float16) for all tensors. This may be much
        faster on GPUs, but has reduced precision and may more often
        cause numerical instability. Only recommended on GPUs.
        """
        if cls.device.type == 'cpu':
            msg = "WARNING: half precision not recommend on CPU"
            _log.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        cls.dtype = torch.float16

    @classmethod
    def use_single_precision(cls):
        """
        Use single precision (float32) for all tensors. This may be
        faster on GPUs, but has reduced precision and may more often
        cause numerical instability.
        """
        cls.dtype = torch.float32

    @classmethod
    def use_double_precision(cls):
        """
        Use double precision (float64) for all tensors. This is the
        recommended precision for numerical stability, but can be
        significantly slower.
        """
        cls.dtype = torch.float64

    @classmethod
    def use_cpu(cls, n=None):
        """
        Use the CPU instead of the GPU for tensor calculations. This is
        the default if no GPU is available. If you have more than one
        CPU, you can use a specific CPU by setting `n`.
        """
        if n is None:
            cls.device = torch.device('cpu')
        else:
            cls.device = torch.device('cpu', n)

    @classmethod
    def use_gpu(cls, n: t.Optional[int] = None):
        """
        Use the GPU instead of the CPU for tensor calculations. This is
        the default if a GPU is available. If you have more than one
        GPU, you can use a specific GPU by setting ``n``.
        """
        if not torch.cuda.is_available():
            msg = "CUDA is not available"
            _log.error(msg)
            raise CudaNotAvailableError(msg)

        if n is not None:
            if not isinstance(n, int):
                msg = "The GPU number must be an integer"
                _log.exception(msg)
                raise TypeError(msg)
            if n < 0:
                msg = "The GPU number must be a non-negative integer"
                _log.exception(msg)
                raise ValueError(msg)
            if torch.cuda.device_count() <= n:
                msg = f"CUDA GPU '{n}' is not available"
                _log.error(msg)
                raise CudaNotAvailableError(msg, n)

            cls.device = torch.device('cuda', n)

        cls.device = torch.device('cuda', torch.cuda.current_device())

    @classmethod
    def print_gpu_information(cls):
        """
        Print information about whether CUDA is supported, and if so
        which GPU is being used.
        """
        if not torch.cuda.is_available():
            msg = "CUDA is not available"
            print(msg)
            return

        print("CUDA is available:")
        current = None
        if cls.device.type == 'cuda':
            current = cls.device.index
        for n in range(torch.cuda.device_count()):
            print(
                "%2d  %s%s" % (
                    n, torch.cuda.get_device_name(n),
                    " (selected)" if n == current else "")
            )

    @classmethod
    def set_eps(cls, val):
        """
        Set the positive minimum for kernel parameters. This is usually
        slightly larger than zero to avoid numerical instabilities.
        """
        cls.eps = val

    def __repr__(self):
        to_return = self.__class__.__name__ + "("

        to_return += f"dtype={self.dtype}, "
        to_return += f"device={self.device}, "
        to_return += f"eps={self.eps:.2e}, "

        if to_return.endswith(", "):
            to_return = to_return[:-2]
        to_return += ")"

        return to_return


config = Config()
conf = config  # Alias for config


# noinspection PyMissingOrEmptyDocstring
def use_half_precision():
    conf.use_half_precision()


# noinspection PyMissingOrEmptyDocstring
def use_single_precision():
    conf.use_single_precision()


# noinspection PyMissingOrEmptyDocstring
def use_double_precision():
    conf.use_double_precision()


# noinspection PyMissingOrEmptyDocstring
def use_cpu(n=None):
    conf.use_cpu(n)


# noinspection PyMissingOrEmptyDocstring
def use_gpu(n=None):
    conf.use_gpu(n)


# noinspection PyMissingOrEmptyDocstring
def print_gpu_information():
    conf.print_gpu_information()


# noinspection PyMissingOrEmptyDocstring
def set_eps(val):
    conf.set_eps(val)


use_half_precision.__doc__ = Config.use_half_precision.__doc__
use_single_precision.__doc__ = Config.use_single_precision.__doc__
use_double_precision.__doc__ = Config.use_double_precision.__doc__
use_cpu.__doc__ = Config.use_cpu.__doc__
use_gpu.__doc__ = Config.use_gpu.__doc__
print_gpu_information.__doc__ = Config.print_gpu_information.__doc__
set_eps.__doc__ = Config.set_eps.__doc__
