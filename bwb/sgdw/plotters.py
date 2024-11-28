"""
Module to plot the steps of the SGDW algorithm. This module defines
many classes that wrap the SGDW algorithm and plot the steps at
a specific iteration.
"""
import abc
from itertools import product
from typing import Callable, final, Optional, override

import numpy as np
import torch
from matplotlib import pyplot as plt

import bwb.logging_ as logging
from bwb.distributions import DistributionDraw
from bwb.sgdw import wrappers as W
from bwb.sgdw.sgdw import Runnable, SGDW

_log = logging.get_logger(__name__)

type DistDrawPosWgtT = torch.Tensor


class Plotter[DistributionT, PosWgtT](
    Runnable[DistributionT],
    metaclass=abc.ABCMeta
):
    fig: Optional[plt.Figure]
    ax: Optional[plt.Axes]
    sgdw: SGDW[DistributionT, PosWgtT]

    def __init__(
        self,
        sgdw: SGDW[DistributionT, PosWgtT],
        plot_every: Optional[int] = None,
        n_cols=12,
        n_rows=2,
        factor=1.5,
        cmap="binary",
        **kwargs,
    ):
        # Classes
        self.sgdw = sgdw

        # Plotting options
        self.plot_every = plot_every
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.factor = factor
        self.cmap = cmap
        self.plot_kwargs = kwargs

        # Matplotlib objects
        self.fig = None
        self.ax = None

    @property
    def create_distr(self) -> Callable[[PosWgtT], DistributionT]:
        """
        Create a distribution from the position and weight.
        """
        return self.sgdw.as_distribution

    @property
    def k(self) -> int:
        """
        Get the current iteration.
        """
        return self.sgdw.iter_params.k

    @abc.abstractmethod
    def plot(
        self,
        init: Optional[int] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
        """
        Plot the distributions.

        :param init: The initial step to plot.
        :param n_rows: The number of rows.
        :param n_cols: The number of columns.
        :return: The figure and the axes.
        """
        pass

    @final
    def callback(self, x: dict) -> None:
        """
        Callback function to plot the distributions.
        """
        if self.plot_every is None:
            return None

        if self.k % self.plot_every == self.plot_every - 1:
            self.fig, self.ax = self.plot(**self.plot_kwargs)

    @final
    @override
    def run(self) -> DistributionT:
        _log.info("Running the SGDW algorithm from the Plotter")
        self.sgdw.callback = self.callback

        # noinspection PyTypeChecker
        return self.sgdw.run()


# TODO: REFACTORIZAR LAS DOS CLASES DE ABAJO, SE PUEDE ABSTRAER VARIAS COSAS
class PlotterComparison(Plotter[DistributionDraw, DistDrawPosWgtT]):
    sgdw: SGDW[DistributionDraw, DistDrawPosWgtT]

    def __init__(
        self,
        sgdw: SGDW[DistributionDraw, DistDrawPosWgtT],
        plot_every: Optional[int] = None,
        n_cols=12,
        n_rows=1,
        factor=1.5,
        cmap="binary",
        **kwargs,
    ):
        super().__init__(sgdw, plot_every, n_cols, n_rows, factor, cmap, **kwargs)
        if plot_every is not None and plot_every < n_rows * n_cols:
            logging.raise_error(
                "'plot_every' should not be less than n_rows * n_cols."
                f" Currently: {plot_every = } < {n_rows * n_cols = }",
                _log, ValueError
            )
        sgdw = self.sgdw
        self.pos_wgt = sgdw = W.LogPosWgtIterProxy(sgdw)
        self.pos_wgt_samp = sgdw = W.LogPosWgtSampledProxy(sgdw)
        self.sgdw = sgdw

    @final
    @override
    @logging.register_total_time_method(_log)
    def plot(
        self,
        init: int = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
        n_rows = self.n_rows if n_rows is None else n_rows
        n_cols = self.n_cols if n_cols is None else n_cols

        max_imgs = n_rows * n_cols
        max_k = self.k
        init = max_k - max_imgs + 1 if init is None else init
        if init < 0:
            logging.raise_error(
                f"init should be greater than 0. Currently: {init = }",
                _log, ValueError
            )
        if init > max_k - max_imgs + 1:
            logging.raise_error(
                f"init should be less than {max_k - max_imgs + 1}. "
                f"Currently: {init = }",
                _log, ValueError
            )

        row, col = n_rows * 2, n_cols

        fig, ax = plt.subplots(
            row, col, figsize=(col * self.factor, row * self.factor),
            subplot_kw={"xticks": [], "yticks": []}
        )

        fig.suptitle(kwargs.get("title", "SGDW"))

        for i, j in product(range(n_rows), range(n_cols)):
            k = init + j + i * n_cols
            gamma_k = self.sgdw.schd.step_schedule(k)

            ax0, ax1 = ax[i * 2, j], ax[i * 2 + 1, j]

            # Label the y-axis
            if j == 0:
                ax0.set_ylabel("Sample")
                ax1.set_ylabel("Step")

            # Plot the sample
            fig_sample = self.create_distr(self.pos_wgt_samp[k][0])
            ax0.imshow(fig_sample.image, cmap=self.cmap)
            ax0.set_title(f"$k={k}$\n"
                          + f"$\\gamma_k={gamma_k * 100:.1f}\\%$",
                          size="x-small")

            # Plot the step
            fig_step = self.create_distr(self.pos_wgt[k])
            ax1.imshow(fig_step.image, cmap=self.cmap)

        plt.tight_layout(pad=0.3)

        plt.show()

        return fig, ax


class PlotterComparisonProjected(Plotter[DistributionDraw, DistDrawPosWgtT]):
    sgdw: SGDW[DistributionDraw, DistDrawPosWgtT]

    def __init__(
        self,
        sgdw: SGDW[DistributionDraw, DistDrawPosWgtT],
        projector: W.ProjectorFn[DistDrawPosWgtT],
        proj_every: int = 1, *,
        plot_every: Optional[int] = None,
        n_cols=12,
        n_rows=1,
        factor=1.5,
        cmap="binary",
        proj_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(sgdw, plot_every, n_cols, n_rows, factor, cmap, **kwargs)
        if plot_every is not None and plot_every < n_rows * n_cols:
            logging.raise_error(
                "'plot_every' should not be less than n_rows * n_cols."
                f" Currently: {plot_every = } < {n_rows * n_cols = }",
                _log, ValueError
            )
        sgdw = self.sgdw
        self.pos_wgt_samp = sgdw = W.LogPosWgtSampledProxy(sgdw)
        self.pos_wgt = sgdw = W.LogPosWgtIterProxy(sgdw)
        proj_kwargs_: dict = proj_kwargs if proj_kwargs is not None else {}
        self.sgdw_proj = sgdw = W.SGDWProjectedDecorator(sgdw, projector, proj_every, **proj_kwargs_)
        self.pos_wgt_proj = sgdw = W.LogPosWgtIterProxy(sgdw)
        self.sgdw = sgdw

    @final
    @override
    @logging.register_total_time_method(_log)
    def plot(
        self,
        init: Optional[int] = None,
        n_rows: Optional[int] = None,
        n_cols: Optional[int] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
        create_distr = self.sgdw.as_distribution
        create_distr: Callable[[DistDrawPosWgtT], DistributionDraw]

        n_rows = self.n_rows if n_rows is None else n_rows
        n_cols = self.n_cols if n_cols is None else n_cols

        max_imgs = n_rows * n_cols
        max_k = self.k
        init = max_k - max_imgs + 1 if init is None else init
        if init < 0:
            logging.raise_error(
                f"init should be greater than 0. Currently: {init = }",
                _log, ValueError
            )
        if init > max_k - max_imgs + 1:
            logging.raise_error(
                f"init should be less than {max_k - max_imgs + 1}. "
                f"Currently: {init = }",
                _log, ValueError
            )

        row, col = n_rows * 3, n_cols

        fig, ax = plt.subplots(
            row, col, figsize=(col * self.factor, row * self.factor),
            subplot_kw={"xticks": [], "yticks": []}
        )

        fig.suptitle(kwargs.get("title", "SGDW"))

        for i, j in product(range(n_rows), range(n_cols)):
            k = init + j + i * n_cols
            gamma_k = self.sgdw.schd.step_schedule(k)

            ax0, ax1, ax2 = ax[i * 3, j], ax[i * 3 + 1, j], ax[i * 3 + 2, j]

            # Label the y-axis
            if j == 0:
                ax0.set_ylabel("Sample")
                ax1.set_ylabel("Step")
                ax2.set_ylabel("Projected")

            # Plot the sample
            fig_sample = create_distr(self.pos_wgt_samp[k][0])
            ax0.imshow(fig_sample.image, cmap=self.cmap)
            ax0.set_title(f"$k={k}$\n"
                          + f"$\\gamma_k={gamma_k * 100:.1f}\\%$",
                          size="x-small")

            # Plot the step
            fig_step = create_distr(self.pos_wgt[k])
            ax1.imshow(fig_step.image, cmap=self.cmap)

            # Plot the projected
            fig_proj = create_distr(self.pos_wgt_proj[k])
            ax2.imshow(fig_proj.image, cmap=self.cmap)

        plt.tight_layout(pad=0.3)

        plt.show()

        return fig, ax
