from __future__ import annotations

import typing

import numpy as np
# noinspection PyPackageRequirements
import ot

from bwb import distributions as distrib
from bwb import logging

_log = logging.get_logger(__name__)


class FitWithDistribution(typing.Protocol):
    """
    Mixing class to be inherited by the BaseTransport class and be fitted with discrete
    distributions defined in the package.
    """

    def fit(self, Xs=None, mu_s=None, ys=None, Xt=None, mu_t=None, yt=None) -> object:
        ...

    @logging.register_init_method(_log)
    def fit_wd(
            self,
            dd_s: distrib.DiscreteDistribution,
            dd_t: distrib.DiscreteDistribution,
    ):
        # noinspection PyPep8Naming
        Xs, Xt = dd_s.enumerate_nz_support_(), dd_t.enumerate_nz_support_()
        mu_s, mu_t = dd_s.nz_probs, dd_t.nz_probs

        return self.fit(Xs=Xs, mu_s=mu_s, Xt=Xt, mu_t=mu_t)


# noinspection PyUnresolvedReferences,PyAttributeOutsideInit,PyPep8Naming,PyUnusedLocal
class BaseTransport(ot.utils.BaseEstimator, FitWithDistribution):
    """Base class for OTDA objects

    .. note::
        All estimators should specify all the parameters that can be set
        at the class level in their ``__init__`` as explicit keyword
        arguments (no ``*args`` or ``**kwargs``).

    The fit method should:

    - estimate a cost matrix and store it in a `cost_` attribute
    - estimate a coupling matrix and store it in a `coupling_` attribute
    - estimate distributions from source and target data and store them in
      `mu_s` and `mu_t` attributes
    - store `Xs` and `Xt` in attributes to be used later on in `transform` and
      `inverse_transform` methods

    `transform` method should always get as input a `Xs` parameter

    `inverse_transform` method should always get as input a `Xt` parameter

    `transform_labels` method should always get as input a `ys` parameter

    `inverse_transform_labels` method should always get as input a `yt` parameter
    """
    metric: str
    norm: str
    limit_max: int
    distribution_estimation: typing.Callable

    @logging.register_total_time_method(_log)
    def fit(self, Xs=None, mu_s=None, ys=None, Xt=None, mu_t=None, yt=None) -> object:
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        mu_s: array-like, shape (n_source_samples,)
            The samples weights.
        ys : array-like, shape (n_source_samples,)
            The training class labels.
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        mu_t: array-like, shape (n_target_samples,)
            The target weights.
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
        nx = self._get_backend(Xs, mu_s, ys, Xt, mu_t, yt)

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = ot.utils.dist(Xs, Xt, metric=self.metric)
            self.cost_ = ot.utils.cost_normalization(self.cost_, self.norm)

            if (ys is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * nx.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in nx.unique(ys) if c != -1]
                for c in classes:
                    idx_s = nx.where((ys != c) & (ys != -1))
                    idx_t = nx.where(yt == c)

                    # all the coefficients corresponding to a source sample
                    # and a target sample :
                    # with different labels get an infinite
                    for j in idx_t[0]:
                        self.cost_[idx_s[0], j] = self.limit_max

            # distribution estimation
            self.mu_s = mu_s if mu_s is not None else self.distribution_estimation(Xs)
            self.mu_t = mu_t if mu_t is not None else self.distribution_estimation(Xt)

            # store arrays of samples
            self.xs_ = Xs
            self.xt_ = Xt

        return self

    def fit_transform(self, Xs=None, mu_s=None, ys=None, Xt=None, mu_t=None, yt=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`
        and transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        mu_s: array-like, shape (n_source_samples,)
            The samples weights.
        ys : array-like, shape (n_source_samples,)
            The class labels for training samples
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        mu_t: array-like, shape (n_target_samples,)
            The target weights.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The source samples.
        """

        return self.fit(Xs, mu_s, ys, Xt, mu_t, yt).transform(Xs, ys, Xt, yt)

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        r"""Transports source samples :math:`\mathbf{X_s}` onto target ones :math:`\mathbf{X_t}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for source samples
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels for target. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xs=Xs):

            if nx.array_equal(self.xs_, Xs):

                # perform standard barycentric mapping
                transp = self.coupling_ / nx.sum(self.coupling_, axis=1)[:, None]

                # set nans to 0
                transp[~ nx.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = nx.dot(transp, self.xt_)
            else:
                # perform out of sample mapping
                indices = nx.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = ot.utils.dist(Xs[bi], self.xs_)
                    idx = nx.argmin(D0, axis=1)

                    # transport the source samples
                    transp = self.coupling_ / nx.sum(
                        self.coupling_, axis=1)[:, None]
                    transp[~ nx.isfinite(transp)] = 0
                    transp_Xs_ = nx.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = nx.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        r"""Propagate source labels :math:`\mathbf{y_s}` to obtain estimated target labels as in
        :ref:`[27] <references-basetransport-transform-labels>`.

        Parameters
        ----------
        ys : array-like, shape (n_source_samples,)
            The source class labels

        Returns
        -------
        transp_ys : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.


        .. _references-basetransport-transform-labels:
        References
        ----------
        .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
           "Optimal transport for multi-source domain adaptation under target shift",
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if ot.utils.check_params(ys=ys):

            ysTemp = ot.utils.label_normalization(nx.copy(ys))
            classes = nx.unique(ysTemp)
            n = len(classes)
            D1 = nx.zeros((n, len(ysTemp)), type_as=self.coupling_)

            # perform label propagation
            transp = self.coupling_ / nx.sum(self.coupling_, axis=0)[None, :]

            # set nans to 0
            transp[~ nx.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ysTemp == c] = 1

            # compute propagated labels
            transp_ys = nx.dot(D1, transp)

            return transp_ys.T

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        r"""Transports target samples :math:`\mathbf{X_t}` onto source samples :math:`\mathbf{X_s}`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The source class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The target class labels. If some target samples are unlabelled, fill the
            :math:`\mathbf{y_t}`'s elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if ot.utils.check_params(Xt=Xt):

            if nx.array_equal(self.xt_, Xt):

                # perform standard barycentric mapping
                transp_ = self.coupling_.T / nx.sum(self.coupling_, 0)[:, None]

                # set nans to 0
                transp_[~ nx.isfinite(transp_)] = 0

                # compute transported samples
                transp_Xt = nx.dot(transp_, self.xs_)
            else:
                # perform out of sample mapping
                indices = nx.arange(Xt.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xt = []
                for bi in batch_ind:
                    D0 = ot.utils.dist(Xt[bi], self.xt_)
                    idx = nx.argmin(D0, axis=1)

                    # transport the target samples
                    transp_ = self.coupling_.T / nx.sum(
                        self.coupling_, 0)[:, None]
                    transp_[~ nx.isfinite(transp_)] = 0
                    transp_Xt_ = nx.dot(transp_, self.xs_)

                    # define the transported points
                    transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self.xt_[idx, :]

                    transp_Xt.append(transp_Xt_)

                transp_Xt = nx.concatenate(transp_Xt, axis=0)

            return transp_Xt

    def inverse_transform_labels(self, yt=None):
        r"""Propagate target labels :math:`\mathbf{y_t}` to obtain estimated source labels
        :math:`\mathbf{y_s}`

        Parameters
        ----------
        yt : array-like, shape (n_target_samples,)

        Returns
        -------
        transp_ys : array-like, shape (n_source_samples, nb_classes)
            Estimated soft source labels.
        """
        nx = self.nx

        # check the necessary inputs parameters are here
        if ot.utils.check_params(yt=yt):

            ytTemp = ot.utils.label_normalization(nx.copy(yt))
            classes = nx.unique(ytTemp)
            n = len(classes)
            D1 = nx.zeros((n, len(ytTemp)), type_as=self.coupling_)

            # perform label propagation
            transp = self.coupling_ / nx.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ nx.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            # compute propagated samples
            transp_ys = nx.dot(D1, transp.T)

            return transp_ys.T


# noinspection PyAttributeOutsideInit
class SinkhornTransport(BaseTransport):
    """Domain Adaptation OT method based on Sinkhorn Algorithm

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=1000)
        The minimum number of iteration before stopping the optimization
        algorithm if it has not converged
    tol : float, optional (default=10e-9)
        The precision required to stop the optimization algorithm.
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : int, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently, the only possible option is
        "ferradans" which uses the method proposed in :ref:`[6] <references-sinkhorntransport>`.
    limit_max: float, optional (default=np.infty)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit a cost defined
        by this variable

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling
    log_ : dictionary
        The dictionary of log, empty dict if parameter log is not True


    .. _references-sinkhorntransport:
    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
           "Optimal Transport for Domain Adaptation," in IEEE Transactions
           on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal
           Transport, Advances in Neural Information Processing Systems (NIPS)
           26, 2013

    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.
    """

    def __init__(self, reg_e=1., method="sinkhorn", max_iter=1000,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean", norm="max",
                 distribution_estimation=ot.da.distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=np.infty):
        self.reg_e = reg_e
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.norm = norm
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map

    @logging.register_total_time_method(_log)
    def fit(self, Xs=None, mu_s=None, ys=None, Xt=None, mu_t=None, yt=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        mu_s: array-like, shape (n_source_samples,)
            The samples weights.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        mu_t: array-like, shape (n_target_samples,)
            The target weights.
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

        super(SinkhornTransport, self).fit(Xs, mu_s, ys, Xt, mu_t, yt)

        # coupling estimation
        returned_ = ot.bregman.sinkhorn(
            a=self.mu_s, b=self.mu_t, M=self.cost_, reg=self.reg_e,
            method=self.method, numItermax=self.max_iter, stopThr=self.tol,
            verbose=self.verbose, log=self.log)

        # deal with the value of log
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_, self.log_ = returned_, dict()

        return self


# noinspection PyAttributeOutsideInit
class EMDTransport(BaseTransport):
    """Domain Adaptation OT method based on Earth Mover's Distance

    Parameters
    ----------
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default="max")
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    log : int, optional (default=False)
        Controls the logs of the optimization algorithm
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently, the only possible option is
        "ferradans" which uses the method proposed in :ref:`[6] <references-emdtransport>`.
    limit_max: float, optional (default=10)
        Controls the semi supervised mode. Transport between labeled source
        and target samples of different classes will exhibit an infinite cost
        (10 times the maximum value of the cost matrix)
    max_iter : int, optional (default=100000)
        The maximum number of iterations before stopping the optimization
        algorithm if it has not converged.

    Attributes
    ----------
    coupling_ : array-like, shape (n_source_samples, n_target_samples)
        The optimal coupling


    .. _references-emdtransport:
    References
    ----------
    .. [1] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
        "Optimal Transport for Domain Adaptation," in IEEE Transactions
        on Pattern Analysis and Machine Intelligence , vol.PP, no.99, pp.1-1
    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
        Regularized discrete optimal transport. SIAM Journal on Imaging
        Sciences, 7(3), 1853-1882.
    """

    def __init__(self, metric="sqeuclidean", norm=None, log=False,
                 distribution_estimation=ot.da.distribution_estimation_uniform,
                 out_of_sample_map='ferradans', limit_max=10,
                 max_iter=100000):
        self.metric = metric
        self.norm = norm
        self.log = log
        self.limit_max = limit_max
        self.distribution_estimation = distribution_estimation
        self.out_of_sample_map = out_of_sample_map
        self.max_iter = max_iter

    @logging.register_total_time_method(_log)
    def fit(self, Xs=None, mu_s=None, ys=None, Xt=None, mu_t=None, yt=None):
        r"""Build a coupling matrix from source and target sets of samples
        :math:`(\mathbf{X_s}, \mathbf{y_s})` and :math:`(\mathbf{X_t}, \mathbf{y_t})`

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        mu_s: array-like, shape (n_source_samples,)
            The samples weights.
        ys : array-like, shape (n_source_samples,)
            The class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        mu_t: array-like, shape (n_target_samples,)
            The target weights.
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

        super(EMDTransport, self).fit(Xs, mu_s, ys, Xt, mu_t, yt)

        returned_ = ot.lp.emd(
            a=self.mu_s, b=self.mu_t, M=self.cost_, numItermax=self.max_iter,
            log=self.log)

        # coupling estimation
        if self.log:
            self.coupling_, self.log_ = returned_
        else:
            self.coupling_, self.log_ = returned_, dict()
        return self
