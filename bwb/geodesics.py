import abc
from typing import Protocol

import torch

import bwb.transports as tpt
from bwb import logging
from bwb.validation import check_is_fitted

_log = logging.get_logger(__name__)


class Geodesic(Protocol):
    def fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        ...

    def interpolate(self, t: float):
        ...


class BaseGeodesic(tpt.FitWithDistribution, metaclass=abc.ABCMeta):
    transport: tpt.BaseTransport

    def __init__(self, transport: tpt.BaseTransport):
        self.transport = transport

    @abc.abstractmethod
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        pass

    @logging.register_total_time(_log)
    @logging.register_init_method(_log)
    def fit(
            self,
            Xs=None, mu_s=None,
            Xt=None, mu_t=None,
    ):
        self.transport.fit(Xs=Xs, mu_s=mu_s, Xt=Xt, mu_t=mu_t)
        self._fit(Xs, mu_s, Xt, mu_t)
        return self

    @abc.abstractmethod
    def interpolate(self, t: float):
        pass


class McCannGeodesic(BaseGeodesic):
    # noinspection PyUnresolvedReferences
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        n, dim_0 = Xs.shape
        m, dim_1 = Xt.shape

        if dim_0 != dim_1:
            raise ValueError(
                f"The number of dimensions between the arrays 'Xs' and 'Xt' do not match. {dim_0} != {dim_1}")

        self.X0_: torch.Tensor = torch.unsqueeze(Xs, dim=1)
        self.X1_: torch.Tensor = torch.unsqueeze(Xt, dim=0)
        self.coupling_: torch.Tensor = self.transport.coupling_

        return self

    @logging.register_total_time(_log)
    def interpolate(self, t: float, rtol=1e-4, atol=1e-6):
        _log.debug(f"Interpolating with {t=:.2f}")
        check_is_fitted(self)
        X = (1 - t) * self.X0_ + t * self.X1_
        coupling = self.coupling_
        nz_coord = torch.nonzero(
            ~torch.isclose(torch.zeros_like(coupling), coupling, rtol=rtol, atol=atol),
            as_tuple=True
        )
        X_to_return = X[nz_coord]
        weights = coupling[nz_coord]
        weights_to_return = weights / weights.sum()
        return X_to_return, weights_to_return


class BarycentricProjGeodesic(BaseGeodesic):
    def _fit(self, Xs=None, mu_s=None, Xt=None, mu_t=None):
        self.X0_ = Xs
        self.mu_0_ = mu_s

        return self

    @logging.register_total_time(_log)
    def interpolate(self, t: float):
        _log.debug(f"Interpolating with {t=:.2f}")
        check_is_fitted(self)
        X_ = (1 - t) * self.X0_ + t * self.transport.transform(self.X0_)

        return X_, self.mu_0_
