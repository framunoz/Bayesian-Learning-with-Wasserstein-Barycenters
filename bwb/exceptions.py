__all__ = [
    "NotFittedError",
    "AutocorrError",
]


# Obtained from sklearn API:
#  https://github.com/scikit-learn/scikit-learn/blob/9aaed498795f68e5956ea762fef9c440ca9eb239/sklearn/exceptions.py#L19
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility."""

    ...


class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time.

    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.

    """

    def __init__(self, tau, *args, **kwargs):
        self.tau = tau
        super(AutocorrError, self).__init__(*args, **kwargs)
