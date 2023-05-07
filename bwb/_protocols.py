from __future__ import annotations

from typing import Protocol

from ot import da as da

from . import _logging
from .distributions import DiscreteDistribution

_log = _logging.get_logger(__name__)


class FitWithMeasure(Protocol):
    """
    Protocol that provides the necessary interface to classes that can be fitted with a specific measure.
    """

    def fit_wm(self, Xs=None, ys=None, mu_s=None, Xt=None, yt=None, mu_t=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The training class labels
        mu_s : (n_source_samples,) array-like, float
            Source histogram (uniform weight if empty list)
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        mu_t : (n_target_samples,) array-like, float
            Target histogram (uniform weight if empty list)

        Returns
        -------
        self : object
            Returns self.
        """
        ...


class FitWithDistribution(FitWithMeasure, Protocol):

    def fit_wd(
            self,
            dd_s: DiscreteDistribution = None, ys=None,
            dd_t: DiscreteDistribution = None, yt=None,
    ):
        _log.debug(f"Fitting with distribution from the FitWithDistribution Protocol.")
        # noinspection PyPep8Naming
        Xs, Xt = dd_s.enumerate_nz_support_(), dd_t.enumerate_nz_support_()
        mu_s, mu_t = dd_s.nz_probs, dd_t.nz_probs

        return self.fit_wm(Xs, ys, mu_s, Xt, yt, mu_t)


class BaseTransport(da.BaseTransport, FitWithDistribution):
    ...


class Transport(Protocol):
    def fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        ...
