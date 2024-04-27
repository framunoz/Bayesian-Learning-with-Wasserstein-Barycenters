import collections as c
import typing as t
from pathlib import Path

import numpy as np
import PIL.Image
import torch

__all__ = [
    "array_like_t",
    "device_t",
    "seed_t",
    "path_t",
    "DiscreteDistribSamplerP",
    "DrawP",
]

type array_like_t = np.ndarray | torch.Tensor | t.Iterable
type device_t = str | torch.device | int | None
type seed_t = int | torch.Generator | None
type path_t = str | Path


@t.runtime_checkable
class DiscreteDistribSamplerP(t.Protocol):
    """
    Protocol for the distribution sampler.
    """
    samples_counter: c.Counter[int]


@t.runtime_checkable
class DrawP(t.Protocol):
    """
    Protocol for the draw.
    """
    image: PIL.Image.Image
