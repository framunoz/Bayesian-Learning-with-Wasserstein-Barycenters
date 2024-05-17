import abc
import typing as t
from itertools import product

import numpy as np
import torch
from matplotlib import pyplot as plt

import bwb.logging_ as logging
from . import wrappers as W
from .sgdw import Runnable, SGDW
from ..distributions import DistributionDraw

_log = logging.get_logger(__name__)


class Plotter[DistributionT, PosWgtT](
    Runnable[DistributionT],
    metaclass=abc.ABCMeta
):
    fig: t.Optional[plt.Figure]
    ax: t.Optional[plt.Axes]
    sgdw: SGDW[DistributionT, PosWgtT]

    def __init__(
        self,
        sgdw: SGDW[DistributionT, PosWgtT],
        plot_every: t.Optional[int] = None,
        n_cols=12,
        n_rows=2,
        factor=1.5,
        cmap="binary"
    ):
        # Classes
        self.sgdw = sgdw

        # Plotting options
        self.plot_every = plot_every
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.factor = factor
        self.cmap = cmap

        # Matplotlib objects
        self.fig = None
        self.ax = None

    @property
    def create_distr(self) -> t.Callable[[PosWgtT], DistributionT]:
        """
        Create a distribution from the position and weight.
        """
        return self.sgdw.create_distribution

    @abc.abstractmethod
    def plot(
        self,
        init: t.Optional[int] = None
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
        """
        Plot the distributions.

        :param init: The initial step to plot.
        :return: The figure and the axes.
        """
        pass

    @t.final
    def callback(self) -> None:
        """
        Callback function to plot the distributions.
        """
        if self.plot_every is None:
            return None

        k = self.sgdw.iter_params.k

        if k % self.plot_every == self.plot_every - 1:
            self.fig, self.ax = self.plot()

    @t.final
    @t.override
    def run(self) -> DistributionT:
        _log.info("Running the SGDW algorithm from the Plotter")
        self.sgdw.callback = self.callback

        # noinspection PyTypeChecker
        return self.sgdw.run()


class PlotterComparison(Plotter[DistributionDraw, torch.Tensor]):
    sgdw: SGDW[DistributionDraw, torch.Tensor]

    def __init__(
        self,
        sgdw: SGDW[DistributionDraw, torch.Tensor],
        plot_every: t.Optional[int] = None,
        n_cols=12,
        n_rows=1,
        factor=1.5,
        cmap="binary"
    ):
        super().__init__(sgdw, plot_every, n_cols, n_rows, factor, cmap)
        if plot_every is not None and plot_every < n_rows * n_cols:
            logging.raise_error(
                "'plot_every' should not be less than n_rows * n_cols."
                f" Currently: {plot_every = } < {n_rows * n_cols = }",
                _log, ValueError
            )
        self.pos_wgt = self.sgdw = W.PosWgtIterRegProxy(self.sgdw)
        self.pos_wgt_samp = self.sgdw = W.PosWgtSampledRegProxy(self.sgdw)

    @t.final
    @t.override
    @logging.register_total_time_method(_log)
    def plot(
        self,
        init: int = None
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:

        max_imgs = self.n_rows * self.n_cols
        max_k = self.sgdw.iter_params.k
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

        row, col = self.n_rows * 2, self.n_cols

        fig, ax = plt.subplots(
            row, col, figsize=(col * self.factor, row * self.factor),
            subplot_kw={"xticks": [], "yticks": []}
        )

        fig.suptitle("SGDW")

        for i, j in product(range(self.n_rows), range(self.n_cols)):
            k = init + j + i * self.n_cols
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


class PlotterComparisonProjected(Plotter[DistributionDraw, torch.Tensor]):
    sgdw: SGDW[DistributionDraw, torch.Tensor]

    def __init__(
        self,
        sgdw: SGDW[DistributionDraw, torch.Tensor],
        projector: W.ProjectorFn[torch.Tensor],
        proj_every: int = 1,
        plot_every: t.Optional[int] = None,
        n_cols=12,
        n_rows=1,
        factor=1.5,
        cmap="binary"
    ):
        super().__init__(sgdw, plot_every, n_cols, n_rows, factor, cmap)
        if plot_every is not None and plot_every < n_rows * n_cols:
            logging.raise_error(
                "'plot_every' should not be less than n_rows * n_cols."
                f" Currently: {plot_every = } < {n_rows * n_cols = }",
                _log, ValueError
            )
        sgdw = self.sgdw
        self.pos_wgt_samp = sgdw = W.PosWgtSampledRegProxy(sgdw)
        self.pos_wgt = sgdw = W.PosWgtIterRegProxy(sgdw)
        sgdw = W.SGDWProjectedDecorator(sgdw, projector, proj_every)
        self.pos_wgt_proj = sgdw = W.PosWgtIterRegProxy(sgdw)
        self.sgdw = sgdw

    @t.final
    @t.override
    @logging.register_total_time_method(_log)
    def plot(
        self,
        init: t.Optional[int] = None
    ) -> tuple[plt.Figure, plt.Axes | np.ndarray[plt.Axes]]:
        create_distr = self.sgdw.create_distribution
        create_distr: t.Callable[[torch.Tensor], DistributionDraw]

        max_imgs = self.n_rows * self.n_cols
        max_k = self.sgdw.iter_params.k
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

        row, col = self.n_rows * 3, self.n_cols

        fig, ax = plt.subplots(
            row, col, figsize=(col * self.factor, row * self.factor),
            subplot_kw={"xticks": [], "yticks": []}
        )

        fig.suptitle("SGDW")

        for i, j in product(range(self.n_rows), range(self.n_cols)):
            k = init + j + i * self.n_cols
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
