from __future__ import annotations

import abc

import numpy as np
from ot import emd
from ot.bregman import sinkhorn
from ot.da import SinkhornTransport, EMDTransport

from .distributions import DistributionDraw


class FitWithMeasure(abc.ABC):
    """
    Protocol that provides the necessary interface to classes that can be fitted with a specific measure.
    """

    @abc.abstractmethod
    def fit_wm(self, Xs=None, ys=None, mu_s=None, Xt=None, yt=None, mu_t=None) -> FitWithMeasure:
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
        pass


class FitWithDistribution(FitWithMeasure, abc.ABC):

    # TODO: Cambiar a un método genérico de instancia Distribution (no solo para DistributionDraw)
    def fit_wd(
            self,
            distr_s: DistributionDraw = None,
            ys=None,
            distr_t: DistributionDraw = None,
            yt=None
    ) -> FitWithDistribution:
        Xs, Xt = np.array(distr_s.support), np.array(distr_t.support)
        mu_s, mu_t = np.array(distr_s.weights), np.array(distr_t.weights)

        return self.fit_wm(Xs, ys, mu_s, Xt, yt, mu_t)


class MySinkhornTransport(SinkhornTransport, FitWithDistribution):

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

        super(SinkhornTransport, self).fit(Xs, ys, Xt, yt)

        # nx: Backend = self._get_backend(Xs, ys, Xt, yt)

        # Case where mu_s and mu_t are provided
        if mu_s is not None and mu_t is not None:
            # TODO: Revisar si esto está bien
            self.mu_s = mu_s
            self.mu_t = mu_t

        # coupling estimation
        returned_ = sinkhorn(
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


class MyEMDTransport(EMDTransport, FitWithDistribution):

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

        super(EMDTransport, self).fit(Xs, ys, Xt, yt)

        # Case where mu_s and mu_t are provided
        if mu_s is not None and mu_t is not None:
            # TODO: Revisar si esto está bien
            self.mu_s = mu_s
            self.mu_t = mu_t

        returned_ = emd(
            a=self.mu_s, b=self.mu_t, M=self.cost_, numItermax=self.max_iter,
            log=self.log)

        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_ = returned_
            self.log_ = dict()
        return self
