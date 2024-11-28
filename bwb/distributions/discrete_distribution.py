"""
Models for discrete distributions.
"""
import typing as t
import warnings

import ot.utils
import torch
# noinspection PyPackageRequirements
import torch.distributions
import torchvision.transforms.functional as F
from PIL import Image

import bwb.logging_ as logging
from bwb.config import config
from bwb.distributions.utils import grayscale_parser
from bwb.protocols import HasDeviceDType
from bwb.utils import shape_validation

__all__ = [
    "DistributionP",
    "wass_distance",
    "DiscreteDistribution",
    "DistributionDraw",
]

_log = logging.get_logger(__name__)

type SizeT = torch.Size | t.Tuple[int, ...]


# noinspection PyPropertyDefinition
class DistributionP(t.Protocol):
    """
    Protocol for the discrete distributions, like the classes
    ``DiscreteDistribution`` and ``DistributionDraw``.
    """

    @property
    def original_support(self) -> torch.Tensor:
        """Original support."""
        ...

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the instance."""
        ...

    @property
    def device(self) -> torch.device:
        """device of the instance."""
        ...

    def enumerate_support_(self, expand=True) -> torch.Tensor:
        """Enumerates the original support ``support`` and not its indices."""
        ...

    def sample_(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """Sample from the original support ``support``."""
        ...

    @property
    def nz_logits(self) -> torch.Tensor:
        """Non-zero logits."""
        ...

    @property
    def nz_probs(self) -> torch.Tensor:
        """Non-zero probs."""
        ...

    def enumerate_nz_support(self, expand=True) -> torch.Tensor:
        """Enumerate non-zero support."""
        ...

    def enumerate_nz_support_(self, expand=True) -> torch.Tensor:
        """Enumerate non-zero support using the original support."""
        ...


def wass_distance(
    p: DistributionP,
    q: DistributionP,
    **solve_sample_kwargs,
):
    r"""
    Compute the Wasserstein distance between two distributions using
    the function `ot.solve_sample
    <https://pythonot.github.io/all.html#ot.solve_sample>`_ from the
    package Python Optimal Transport (POT).

    :param p: The first distribution.
    :param q: The second distribution.
    :param solve_sample_kwargs: Additional keyword arguments for the
        function ``ot.solve_sample``. Defaults is an empty dictionary.
        For further information, see `ot.solve_sample
        <https://pythonot.github.io/all.html#ot.solve_sample>`_.
    :return: The Wasserstein distance between the two distributions.
    """
    def get_pos_wgt(d: DistributionP) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the position and weights of the distribution."""
        return d.enumerate_nz_support_().to(d.dtype), d.nz_probs

    x_p, w_p = get_pos_wgt(p)
    x_q, w_q = get_pos_wgt(q)

    res: ot.utils.OTResult = ot.solve_sample(
        x_p, x_q, w_p, w_q,
        **solve_sample_kwargs
    )

    return res.value


# noinspection PyAbstractClass
class DiscreteDistribution(torch.distributions.Categorical, HasDeviceDType):
    """
    Class that represent a discrete distribution, using de
    `Categorical
    <https://pytorch.org/docs/stable/distributions.html#categorical>`_
    class provided by PyTorch.

    :param weights: Array of probabilities. If it is a Tensor type,
        the device shall be fixed by this parameter.
    :param support: Support of the distribution. Optional.
    :param dtype: The dtype of the distribution.
    :param device: The device of the distribution.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        support: t.Optional[torch.Tensor] = None,
        validate_args=None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        """
        Initializer.

        :param weights: Array of probabilities. If it is a Tensor type,
        the device shall be fixed by this parameter.
        :param support: Support of the distribution. Optional.
        :param validate_args:
        """
        dtype = weights.dtype if dtype is None else dtype
        device = weights.device if device is None else device

        # Save weights and support as tensor
        self.weights: torch.Tensor = torch.as_tensor(
            weights, dtype=dtype, device=device
        )

        self._original_support: t.Optional[torch.Tensor] = None
        if support is not None:
            self._original_support: torch.Tensor = torch.as_tensor(
                support, device=device
            )

        if (
            support is not None
            and len(self.weights) != len(self._original_support)
        ):
            raise ValueError(
                "The sizes of weights and support does not coincide:"
                f" {len(self.weights)} != {len(self._original_support)}"
            )

        super().__init__(probs=self.weights, validate_args=validate_args)

        # Non zero mask
        self._nz_mask: torch.Tensor = self.probs.nonzero(as_tuple=True)

    @property
    def original_support(self) -> torch.Tensor:
        """Original support."""
        if self._original_support is None:
            self._original_support = torch.arange(
                len(self.weights), device=self.device
            )

        return self._original_support

    @property
    def dtype(self) -> torch.dtype:
        """dtype of the instance."""
        return self.weights.dtype

    @property
    def device(self) -> torch.device:
        """device of the instance."""
        return self.weights.device

    def enumerate_support_(self, expand=True) -> torch.Tensor:
        """Enumerates the original support ``support`` and not its indices."""
        return self.original_support[self.enumerate_support(expand)]

    def sample_(self, sample_shape=torch.Size([])) -> torch.Tensor:
        """Sample from the original support ``support``."""
        return self.original_support[self.sample(sample_shape)]

    @property
    def nz_logits(self) -> torch.Tensor:
        """Non-zero logits."""
        return self.logits[self._nz_mask]

    @property
    def nz_probs(self) -> torch.Tensor:
        """Non-zero probs."""
        return self.probs[self._nz_mask]

    def enumerate_nz_support(self, expand=True) -> torch.Tensor:
        """Enumerate non-zero support."""
        return self.enumerate_support(expand)[self._nz_mask]

    def enumerate_nz_support_(self, expand=True) -> torch.Tensor:
        """Enumerate non-zero support using the original support."""
        return self.enumerate_support_(expand)[self._nz_mask]


# TODO: Change this name to ImageDistribution
# noinspection PyAbstractClass
class DistributionDraw(DiscreteDistribution):
    """
    Distribution generated by a drawing, which is represented by a
    matrix.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        shape: SizeT,
        support: torch.Tensor = None,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        """
        Initializer.

        :param weights: The weights of the distribution.
        :param support: The positions of the weights.
        :param shape: The shape of the image that represents the
            distribution.
        """
        # Probably this is useless, because the super class already
        # does this
        # TODO: Delete this in other version
        dtype = weights.dtype if dtype is None else dtype
        device = weights.device if device is None else device

        # Get the shape information
        self.shape = shape_validation(shape, n_dim=2)
        self._grayscale = None  # For the cache of the grayscale
        # For the cache of the grayscale weights
        self._grayscale_weights = None

        super(DistributionDraw, self).__init__(
            weights=weights,
            support=support,
            dtype=dtype,
            device=device
        )

    @property
    def original_support(self) -> torch.Tensor:
        """Original support."""
        if self._original_support is None:
            n, m = self.shape
            index = torch.arange(n * m,
                                 device=self.device,
                                 dtype=torch.int64).reshape(-1, 1)
            self._original_support = torch.cat((index // m, index % m), 1)

        return self._original_support

    @classmethod
    def from_weights(
        cls,
        weights: torch.Tensor,
        shape: SizeT,
        device: torch.Tensor = None,
        dtype: torch.dtype = None,
    ) -> t.Self:
        """
        Build an instance from the weights and the shape, assuming that
        all weights correspond to an image coordinate.

        Deprecated: use the constructor instead.

        :param weights: The weights of the distribution.
        :param shape: The shape of the image that represents the
            distribution.
        :param dtype: The dtype of the distribution.
        :param device: The device of the distribution.
        :return: an instance of :py:class:`DistributionDraw`
        """
        warnings.warn(
            "This method is deprecated, use the constructor instead.",
            DeprecationWarning, stacklevel=2
        )
        return cls(weights, shape, device=device, dtype=dtype)

    @classmethod
    def from_discrete_distribution(
        cls,
        dd: DiscreteDistribution,
        shape: SizeT,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> t.Self:
        """
        Build an instance from a discrete distribution and the shape of
        the image.
        """
        return cls(
            support=dd.original_support,
            weights=dd.weights,
            shape=shape,
            device=device,
            dtype=dtype
        )

    @classmethod
    def from_grayscale_weights(
        cls,
        grayscale_weights: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> t.Self:
        """Build an instance from the grayscale weights."""
        device = grayscale_weights.device if device is None else device
        dtype = grayscale_weights.dtype if dtype is None else dtype

        # Save the grayscales for create images
        grayscale_weights: torch.Tensor = torch.as_tensor(
            grayscale_weights,
            dtype=dtype,
            device=device,
        ).squeeze()

        # Get the shape information
        shape = shape_validation(
            grayscale_weights.shape,
            n_dim=2,
            msg="The 'grayscale_weights' tensor must have dimension {n_dim}.",
        )

        grayscale_weights /= torch.sum(grayscale_weights)
        weights = grayscale_weights.reshape((-1,))

        to_return = cls(weights=weights, shape=shape,
                        device=device, dtype=dtype)

        to_return._grayscale_weights = grayscale_weights

        return to_return

    @classmethod
    def from_array(
        cls,
        grayscale: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> t.Self:
        """
        Build an instance from the grayscale image.

        :param dtype: The dtype of the distribution.
        :param device: The device of the distribution.
        :param grayscale: The grayscale image.
        :return: an instance of :py:class:`DistributionDraw`
        """
        device = grayscale.device if device is None else device
        dtype = config.dtype if dtype is None else dtype

        # Save the grayscales for create images
        grayscale: torch.Tensor = torch.as_tensor(
            grayscale,
            dtype=torch.uint8,  # Use uint8 for images
            device=device
        ).squeeze()

        # Get the shape information
        shape = shape_validation(
            grayscale.shape,
            n_dim=2,
            msg="The 'grayscale' tensor must have dimension {n_dim}.",
        )

        # Get the weights from the grayscale
        grayscale_weights: torch.Tensor = grayscale / 255
        grayscale_weights /= torch.sum(grayscale_weights)
        weights = grayscale_weights.reshape((-1,)).to(dtype)

        to_return = cls(weights=weights, shape=shape,
                        device=device, dtype=dtype)

        to_return._grayscale = grayscale
        to_return._grayscale_weights = grayscale_weights.to(dtype)

        return to_return

    @property
    @logging.register_total_time_method(_log)
    def grayscale(self) -> torch.Tensor:
        """A matrix representing the gray scale of the image."""
        if self._grayscale is not None:
            return self._grayscale

        if self._grayscale_weights is not None:
            grayscale = torch.clone(self._grayscale_weights)
            grayscale /= grayscale.max()
            grayscale *= torch.tensor(255,
                                      dtype=self.dtype, device=self.device)
            self._grayscale = torch.as_tensor(
                grayscale, dtype=torch.uint8, device=self.device
            )

        else:
            weights = self.weights
            support = self.original_support
            # Compute the grayscale
            self._grayscale = grayscale_parser(
                self.shape, weights, support,
            )

        return self._grayscale

    @property
    @logging.register_total_time_method(_log)
    def grayscale_weights(self) -> torch.Tensor:
        """
        A matrix representing the gray scale of the image as a
        probability weight.
        """
        if self._grayscale_weights is None:
            gs_weights = self.grayscale.to(self.dtype) / 255
            gs_weights /= torch.sum(gs_weights)
            self._grayscale_weights = gs_weights

        return self._grayscale_weights

    @property
    def image(self) -> Image.Image:
        """
        Representation of the Image.

        :return: A `PIL.Image.Image
            <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image>`_
            object.
        """
        return F.to_pil_image(255 - self.grayscale)

    def __repr__(self) -> str:
        to_return = self.__class__.__name__ + "("
        to_return += f"shape={tuple(self.shape)}, "
        to_return += f"device={self.device}, "
        to_return += f"dtype={self.dtype}), "

        if to_return.endswith(", "):
            to_return = to_return[:-len(", ")]

        to_return += ")"

        return to_return

    def _repr_png_(self) -> bytes:
        """iPython display hook support for PNG format.

        :returns: png version of the image as bytes
        """
        # noinspection PyProtectedMember
        return self.image._repr_png_()
