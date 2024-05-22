import collections as c
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable

import numpy as np
import PIL.Image
import torch

__all__ = [
    "ArrayLikeT",
    "DeviceT",
    "SeedT",
    "PathT",
    "DiscreteDistribSamplerP",
    "DrawP",
]

type ArrayLikeT = np.ndarray | torch.Tensor | Iterable
type DeviceT = str | torch.device | int | None
type SeedT = int | torch.Generator | None
type PathT = str | Path


@runtime_checkable
class DiscreteDistribSamplerP(Protocol):
    """
    Protocol for the distribution sampler.
    """
    samples_counter: c.Counter[int]


@runtime_checkable
class DrawP(Protocol):
    """
    Protocol for the draw.
    """
    image: PIL.Image.Image
