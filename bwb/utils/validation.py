import math
from inspect import isclass

import torch

from bwb.exceptions import NotFittedError

__all__ = [
    "check_is_fitted",
    "shape_validation",
    "shape_weights_validation",
]


def check_is_fitted(estimator, attributes=None, *, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.
    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError with the given message.
    If an estimator does not set any attributes with a trailing
    underscore, it can define a ``__sklearn_is_fitted__`` method
    returning a boolean to specify if the estimator is fitted or not.

    :param estimator: Estimator instance for which the check is performed.
    :param attributes: Attribute name(s) given as string or a list/tuple
        of strings.
    :param msg: The error message.
    :param all_or_any: Specify whether all or any of the given
        attributes must exist.
    :raises TypeError: If the estimator is a class or not an estimator
        instance.
    :raises NotFittedError: If the attributes are not found.
    """
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not hasattr(estimator, "fit"):
        raise TypeError("%s is not an estimator instance." % estimator)

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        fitted = all_or_any([hasattr(estimator, attr) for attr in attributes])
    else:
        fitted = [
            v
            for v in vars(estimator)
            if v.endswith("_") and not v.startswith("__")
        ]

    if not fitted:
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def shape_validation(
    shape, n_dim=2, msg="The shape must have dimension {n_dim}."
) -> torch.Size:
    """
    Function that validates the shape of the tensor.

    :param shape: The shape of the tensor.
    :param n_dim: The number of dimensions of the tensor.
    :param msg: The message to show if the shape is not valid.
    :return: The shape of the tensor.
    """
    if len(shape) != n_dim:
        raise ValueError(msg.format(n_dim=n_dim))
    return torch.Size(shape)


def shape_weights_validation(shape, weights, n_dim=2):
    """
    Function that validates the shape of the tensor and the weights.

    :param shape: The shape of the tensor.
    :param weights: The weights of the tensor.
    :param n_dim: The number of dimensions of the tensor.
    :return: The shape of the tensor.
    """
    shape = shape_validation(shape, n_dim)
    if math.prod(shape) != len(weights):
        raise ValueError(
            "The weights must be equals to the product of the shape, "
            f"where prod(shape)={math.prod(shape)}."
        )
    return shape
