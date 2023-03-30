__all__ = [
    "NotFittedError",
]


# Obtained from sklearn API:
#  https://github.com/scikit-learn/scikit-learn/blob/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/exceptions.py#L19
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility."""
    ...
