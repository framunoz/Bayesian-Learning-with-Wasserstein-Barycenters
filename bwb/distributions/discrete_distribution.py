import torch
from pyro.distributions import Categorical

from bwb.config import Config

config = Config()


# noinspection PyAbstractClass
class DiscreteDistribution(Categorical):
    """
    Class that represent a discrete distribution, using de `Categorical` class provided by Pyro.
    """

    def __init__(
            self,
            pk, xk=None,
            device=None,
            validate_args=None,
    ):
        """
        Initialiser.

        :param pk: Array of probabilities. If it is a Tensor type, the device shall be fixed by this parameter.
        :type pk: array_like
        :param xk: Support of the distribution. Optional.
        :type xk: array_like or None
        :param device: The device of the instance and Tensors. If not provided, it will be inferred by the ``pk``
         Tensor, or by the default configuration.
        :type device: device or str.
        :param validate_args:
        """
        # If pk is a tensor, use the device of that tensor, else, use CUDA if possible
        if isinstance(pk, torch.Tensor):
            device: torch.device = torch.device(device or torch.device(pk.device))
        self.device: torch.device = device or config.device

        # Save pk and xk as tensor
        self.pk = torch.as_tensor(pk, dtype=config.dtype, device=self.device)

        xk = xk if xk is not None else range(len(self.pk))
        self.xk = torch.as_tensor(xk, device=self.device)

        if len(self.pk) != len(self.xk):
            raise ValueError(
                "The sizes of pk and xk does not coincide:"
                f" {len(self.pk)} != {len(self.xk)}")

        super().__init__(probs=self.pk, validate_args=validate_args)

    def enumerate_support_(self, expand=True):
        """Enumerates the original support ``xk`` and not its indices."""
        return self.xk[self.enumerate_support(expand)]

    def sample_(self, sample_shape=torch.Size([])):
        """Sample from the original support ``xk``."""
        return self.xk[self.sample(sample_shape)]
