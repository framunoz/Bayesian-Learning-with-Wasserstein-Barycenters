import abc
from typing import Protocol

import torch

from . import _logging
from ._protocols import BaseTransport, FitWithDistribution
from .distributions import DiscreteDistribution
from .validation import check_is_fitted

_log = _logging.get_logger(__name__)


class Geodesic(Protocol):
    def fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        ...

    def interpolate(self, t: float):
        ...


class BaseGeodesic(FitWithDistribution, metaclass=abc.ABCMeta):
    transport: BaseTransport

    def __init__(self, transport: BaseTransport):
        self.transport = transport

    @abc.abstractmethod
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        pass

    @_logging.register_total_time(_log)
    def fit_wm(
            self,
            Xs=None, ys=None, mu_s=None,
            Xt=None, yt=None, mu_t=None,
    ):
        _log.debug(f"Fitting with measure from the BaseGeodesic class.")
        self.transport.fit_wm(Xs=Xs, ys=ys, mu_s=mu_s, Xt=Xt, yt=yt, mu_t=mu_t)
        self._fit(Xs, mu_s, Xt, mu_t)
        return self

    @_logging.register_total_time(_log)
    def fit(
            self,
            dd_s: DiscreteDistribution = None,
            dd_t: DiscreteDistribution = None,
    ):
        return self.fit_wd(dd_s=dd_s, dd_t=dd_t)

    @abc.abstractmethod
    def interpolate(self, t: float):
        pass


class McCannGeodesic(BaseGeodesic):
    # noinspection PyUnresolvedReferences
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        _log.debug(f"Using the method _fit.")
        n, dim_0 = Xs.shape
        _log.debug(f"{n=}, {dim_0=}")
        m, dim_1 = Xt.shape
        _log.debug(f"{m=}, {dim_1=}")

        if dim_0 != dim_1:
            raise ValueError(
                f"The number of dimensions between the arrays 'Xs' and 'Xt' do not match. {dim_0} != {dim_1}")

        self.X0_: torch.Tensor = torch.unsqueeze(Xs, dim=1)
        self.X1_: torch.Tensor = torch.unsqueeze(Xt, dim=0)
        self.coupling_: torch.Tensor = self.transport.coupling_

        return self

    @_logging.register_total_time(_log)
    def interpolate(self, t: float, rtol=1e-4, atol=1e-6):
        _log.debug(f"Interpolating with {t=:.2f}")
        check_is_fitted(self)
        X = (1 - t) * self.X0_ + t * self.X1_
        _log.debug(f"{X.shape = }")
        coupling = self.coupling_
        _log.debug(f"{coupling.shape = }")
        nz_coord = torch.nonzero(
            ~torch.isclose(torch.zeros_like(coupling), coupling, rtol=rtol, atol=atol),
            as_tuple=True
        )
        _log.debug(f"{nz_coord[0].shape = }")
        X_to_return = X[nz_coord]
        _log.debug(f"{X_to_return.shape = }")
        weights = coupling[nz_coord]
        _log.debug(f"{weights.shape = }")
        _log.debug(f"{weights.sum() = :.4f}")
        weights_to_return = weights / weights.sum()
        return X_to_return, weights_to_return


class BarycentricProjGeodesic(BaseGeodesic):
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        self.X0_ = Xs
        self.mu_0_ = mu_s

        return self

    @_logging.register_total_time(_log)
    def interpolate(self, t: float):
        _log.debug(f"Interpolating with {t=:.2f}")
        check_is_fitted(self)
        X_ = (1 - t) * self.X0_ + t * self.transport.transform(self.X0_)

        return X_, self.mu_0_

    ...
