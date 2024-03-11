"""
Module that implements the missing functionalities of convolutional wasserstein distances.
"""
import warnings

import torch
from ot.backend import get_backend
from bwb.config import config


def _entropy(mu, stabThr=1e-30):
    nx = get_backend(mu)
    return -nx.sum(mu * nx.log(mu + stabThr))


def _entropic_sharpening(mu, H0, stabThr=1e-30):
    beta = 1.0
    if _entropy(mu, stabThr) > H0 + stabThr:
        from xitorch.optimize import rootfinder

        try:
            beta = rootfinder(
                fcn=lambda beta_: _entropy(torch.pow(mu, beta_), stabThr) - H0,
                y0=torch.tensor(1.0, dtype=mu.dtype, device=mu.device),
            )
        except:
            warnings.warn("Rootfinder has failed. Will use beta=1.0")
    if beta < 0.0:
        beta = 1.0

    mu = torch.pow(mu, beta)

    return mu / torch.sum(mu)


def convolutional_barycenter2d(
    A,
    reg,
    weights=None,
    method="sinkhorn",
    numItermax=10000,
    stopThr=1e-8,
    verbose=False,
    log=False,
    warn=True,
    entrop_sharp=False,
    H0=None,
    **kwargs
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions :math:`\mathbf{A}`
    where :math:`\mathbf{A}` is a collection of 2D images. This function use the API used in POT.

     The function solves the following optimization problem:

    .. math::
       \mathbf{a} = \mathop{\arg \min}_\mathbf{a} \quad \sum_i W_{reg}(\mathbf{a},\mathbf{a}_i)

    where :

    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein
      distance (see :py:func:`ot.bregman.sinkhorn`)
    - :math:`\mathbf{a}_i` are training distributions (2D images) in the mast two dimensions
      of matrix :math:`\mathbf{A}`
    - `reg` is the regularization strength scalar value

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm
    as proposed in :ref:`[21] <references-convolutional-barycenter-2d>`

    Parameters
    ----------
    A : array-like, shape (n_hists, width, height)
        `n` distributions (2D images) of size `width` x `height`
    reg : float
        Regularization term >0
    weights : array-like, shape (n_hists,)
        Weights of each image on the simplex (barycentric coordinates)
    method : string, optional
        method used for the solver either 'sinkhorn' or 'sinkhorn_log'
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    stabThr : float, optional
        Stabilization threshold to avoid numerical precision issue
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    warn : bool, optional
        if True, raises a warning if the algorithm doesn't convergence.
    entrop_sharp : bool, optional
        if True, use an entropic regularization.
    H0 : float, optional
        Only used if entrop_sharp is True. Is the threshold of the entropic regularization
        algorithm.

    Returns
    -------
    a : array-like, shape (width, height)
        2D Wasserstein barycenter
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-convolutional-barycenter-2d:
    References
    ----------

    .. [21] Solomon, J., De Goes, F., Peyré, G., Cuturi, M., Butscher,
        A., Nguyen, A. & Guibas, L. (2015).     Convolutional wasserstein distances:
        Efficient optimal transportation on geometric domains. ACM Transactions
        on Graphics (TOG), 34(4), 66

    .. [37] Janati, H., Cuturi, M., Gramfort, A. Proceedings of the 37th
        International Conference on Machine Learning, PMLR 119:4692-4701, 2020
    """

    if method.lower() == "sinkhorn":
        return _convolutional_barycenter2d(
            A,
            reg,
            weights=weights,
            numItermax=numItermax,
            stopThr=stopThr,
            verbose=verbose,
            log=log,
            warn=warn,
            entrop_sharp=entrop_sharp,
            H0=H0,
            **kwargs
        )
    # We will not use the function that uses the logarithm
    else:
        raise ValueError("Unknown method '%s'." % method)


def _convolutional_barycenter2d(
    A,
    reg,
    weights=None,
    entrop_sharp=False,
    H0=None,
    numItermax=10000,
    stopThr=1e-9,
    stabThr=1e-30,
    checkSteps=10,
    verbose=False,
    log=False,
    warn=True,
):
    r"""Compute the entropic regularized wasserstein barycenter of distributions A
    where A is a collection of 2D images.
    """

    if not isinstance(A, torch.Tensor):
        A = torch.stack(A)

    nx = get_backend(A)

    dtype, device = nx.dtype_device(A)

    n_hists, width, height = A.shape

    if weights is None:
        weights = nx.ones((n_hists,), type_as=A)
    else:
        assert len(weights) == n_hists
        weights = torch.as_tensor(weights, dtype=dtype, device=device)

    weights = weights / nx.sum(weights)
    weights_ = weights[:, None, None]

    if entrop_sharp and H0 is None:
        H0 = max([_entropy(mu_i) for mu_i in A])

    if log:
        log = {"err": []}

    bar = nx.ones(A.shape[1:], type_as=A)
    bar /= nx.sum(bar)
    old_bar = bar
    V = nx.ones(A.shape, type_as=A)
    W = nx.ones(A.shape, type_as=A)

    # build the convolution operator
    # this is equivalent to blurring on horizontal then vertical directions
    t = nx.linspace(0, 1, A.shape[1]).to(dtype=dtype, device=device)
    [Y, X] = nx.meshgrid(t, t)
    K1 = nx.exp(-((X - Y) ** 2) / reg)

    t = nx.linspace(0, 1, A.shape[2]).to(dtype=dtype, device=device)
    [Y, X] = nx.meshgrid(t, t)
    K2 = nx.exp(-((X - Y) ** 2) / reg)

    def convol_imgs(imgs):
        kx = nx.einsum("...ij,kjl->kil", K1, imgs)
        kxy = nx.einsum("...ij,klj->kli", K2, kx)
        return kxy

    ii = 0
    for ii in range(numItermax):
        # Project onto C_1
        W = A / torch.maximum(convol_imgs(V), config.eps)
        D = V * convol_imgs(W)
        bar = nx.exp(nx.sum(weights_ * nx.log(D + stabThr), axis=0))

        # The optional entropic regularization
        if entrop_sharp:
            bar = _entropic_sharpening(bar, H0, stabThr=stabThr)

        # Project onto C_2
        V = V * bar[None] / torch.maximum(D, config.eps)

        if ii % checkSteps == 0:
            err = nx.mean(nx.abs(old_bar - bar))
            # log and verbose print
            if log:
                log["err"].append(err)

            if verbose:
                if ii % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(ii, err))

            if err < stopThr:
                break

        old_bar = bar

    else:
        if warn:
            warnings.warn(
                "Convolutional Sinkhorn did not converge. "
                "Try a larger number of iterations `numItermax` "
                "or a larger entropy `reg`."
            )
    if log:
        log["niter"] = ii
        log["V"] = V
        log["W"] = W
        return bar, log
    else:
        return bar
