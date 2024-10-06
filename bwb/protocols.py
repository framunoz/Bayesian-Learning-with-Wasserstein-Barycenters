import abc
import typing as t

import torch


class HasDeviceDTypeP(t.Protocol):
    """
    A protocol for objects that have a dtype and a device.
    """

    @property
    def dtype(self) -> torch.dtype:
        """
        The dtype of the object.

        :return: The dtype of the object.
        """
        ...

    @property
    def device(self) -> torch.device:
        """
        The device of the object.

        :return: The device of the object.
        """
        ...


class HasDeviceDType(metaclass=abc.ABCMeta):
    """
    A base class for objects that have a dtype and a device.
    """

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        """
        The dtype of the object.

        :return: The dtype of the object.
        """
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """
        The device of the object.

        :return: The device of the object.
        """
        pass
