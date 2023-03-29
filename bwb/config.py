import logging
from threading import Lock

import torch

# Get logger

_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

# Configure a formatter
_formatter = logging.Formatter("%(asctime)s: %(levelname)s [%(name)s:%(lineno)s]\n> %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
_log.addHandler(_stream_handler)


class SingletonMeta(type):
    """Metaclass to implements Singleton Pattern. Obtained from
    https://refactoring.guru/design-patterns/singleton/python/example#example-1 """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    # The dtype selected by default: A double
    dtype = torch.float64

    # Use cuda if is possible
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    # Settings for the smallest number
    _eps = None

    @property
    def eps(self):
        if self._eps is None:
            return torch.finfo(self.dtype).eps
        return self._eps

    @eps.setter
    def eps(self, value):
        self._eps = value


config = Config()


def use_half_precision():
    """
    Use half precision (float16) for all tensors. This may be much faster on GPUs, but has reduced precision and may more often cause numerical instability. Only recommended on GPUs.
    """
    if config.device.type == 'cpu':
        print('WARNING: half precision not recommend on CPU')
    config.dtype = torch.float16


def use_single_precision():
    """
    Use single precision (float32) for all tensors. This may be faster on GPUs, but has reduced precision and may more often cause numerical instability.
    """
    config.dtype = torch.float32


def use_double_precision():
    """
    Use double precision (float64) for all tensors. This is the recommended precision for numerical stability, but can be significantly slower.
    """
    config.dtype = torch.float64


def use_cpu(n=None):
    """
    Use the CPU instead of the GPU for tensor calculations. This is the default if no GPU is available. If you have more than one CPU, you can use a specific CPU by setting `n`.
    """
    if n is None:
        config.device = torch.device('cpu')
    else:
        config.device = torch.device('cpu', n)


def use_gpu(n=None):
    """
    Use the GPU instead of the CPU for tensor calculations. This is the default if a GPU is available. If you have more than one GPU, you can use a specific GPU by setting `n`.
    """
    if not torch.cuda.is_available():
        _log.error("CUDA is not available")
    elif n is not None and (not isinstance(n, int) or n < 0 or torch.cuda.device_count() <= n):
        _log.error("CUDA GPU '%s' is not available" % (n,))
    elif n is None:
        config.device = torch.device('cuda', torch.cuda.current_device())
    else:
        config.device = torch.device('cuda', n)


def print_gpu_information():
    """
    Print information about whether CUDA is supported, and if so which GPU is being used.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return

    print("CUDA is available:")
    current = None
    if config.device.type == 'cuda':
        current = config.device.index
    for n in range(torch.cuda.device_count()):
        print("%2d  %s%s" % (n, torch.cuda.get_device_name(n), " (selected)" if n == current else ""))


def set_eps(val):
    """
    Set the positive minimum for kernel parameters. This is usually slightly larger than zero to avoid numerical instabilities. Default is at 1e-8.
    """
    config.eps = val
