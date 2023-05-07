from __future__ import annotations

import ot
import ot.da as da

from bwb import logging
from bwb._protocols import BaseTransport

_log = logging.get_logger(__name__)


class SinkhornTransport(da.SinkhornTransport, BaseTransport):

    @logging.register_total_time(_log)
    def fit_wm(self, Xs=None, ys=None, mu_s=None, Xt=None, yt=None, mu_t=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        _log.debug(f"Adjusting with a measure from the SinkhornTransport class")

        super(da.SinkhornTransport, self).fit(Xs, ys, Xt, yt)

        # nx: Backend = self._get_backend(Xs, ys, Xt, yt)

        # Case where mu_s and mu_t are provided
        if mu_s is not None and mu_t is not None:
            # TODO: Revisar si esto está bien
            self.mu_s = mu_s
            self.mu_t = mu_t

        # coupling estimation
        returned_ = ot.bregman.sinkhorn(
            a=self.mu_s, b=self.mu_t, M=self.cost_, reg=self.reg_e,
            numItermax=self.max_iter, stopThr=self.tol,
            verbose=self.verbose, log=self.log)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()

        return self


class EMDTransport(da.EMDTransport, BaseTransport):

    # noinspection PyAttributeOutsideInit
    @logging.register_total_time(_log)
    def fit_wm(self, Xs=None, ys=None, mu_s=None, Xt=None, yt=None, mu_t=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        _log.debug(f"Adjusting with a measure from the EMDTransport class")
        _log.debug(f"Shapes: {Xs.shape=}, {Xt.shape=}")

        super(da.EMDTransport, self).fit(Xs, ys, Xt, yt)

        # Case where mu_s and mu_t are provided
        if mu_s is not None and mu_t is not None:
            # TODO: Revisar si esto está bien
            self.mu_s = mu_s
            self.mu_t = mu_t

        returned_ = ot.emd(
            a=self.mu_s, b=self.mu_t, M=self.cost_, numItermax=self.max_iter,
            log=self.log)

        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self
