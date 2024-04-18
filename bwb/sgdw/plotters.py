import abc
import typing as t

import torch
from matplotlib import pyplot as plt

from bwb.distributions import DistributionDraw
from bwb.logging import get_logger
from bwb.sgdw.sgdw import BaseSGDW
from bwb.sgdw.sgdw import Runnable
from bwb.sgdw.utils import _PosWgt
from bwb.sgdw.utils import History
from bwb.sgdw.utils import ReportOptions
from bwb.utils import _DistributionT

_log = get_logger(__name__)


class Plotter(Runnable[_DistributionT, _PosWgt], metaclass=abc.ABCMeta):
    fig: t.Optional[plt.Figure]
    ax: t.Optional[plt.Axes]

    def __init__(
        self,
        sgdw: BaseSGDW[_DistributionT, _PosWgt],
        plot_every=100,
        n_cols=12,
        n_rows=2,
        factor=1.5,
        cmap="binary"
    ):
        # Classes
        self.sgdw = sgdw
        self.hist = sgdw.hist

        # Plotting options
        self.plot_every = plot_every
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.factor = factor
        self.cmap = cmap

        # Matplotlib objects
        self.fig = None
        self.ax = None

        # run options to impose the plots
        self.pos_wgt_hist = False
        self.distr_hist = False
        self.pos_wgt_samp_hist = False
        self.distr_samp_hist = False

    @abc.abstractmethod
    def plot(self, init: int = None):
        pass

    def callback(self) -> None:
        k = self.sgdw.iter_params.k

        if k % self.plot_every == self.plot_every - 1:
            self.fig, self.ax = self.plot()

    def run(
        self,
        pos_wgt_hist: bool = False, distr_hist: bool = False,
        pos_wgt_samp_hist: bool = False, distr_samp_hist: bool = False,
        include_dict: t.Optional[ReportOptions] = None
    ) -> t.Union[_DistributionT, tuple[_DistributionT, History[_DistributionT, _PosWgt]]]:
        _log.info("Running the SGDW algorithm from the Plotter")
        self.sgdw.callback = self.callback

        # noinspection PyTypeChecker
        return self.sgdw.run(
            pos_wgt_hist or self.pos_wgt_hist,
            distr_hist or self.distr_hist,
            pos_wgt_samp_hist or self.pos_wgt_samp_hist,
            distr_samp_hist or self.distr_samp_hist,
            include_dict,
        )


class PlotterComparison(Plotter[DistributionDraw, torch.Tensor]):

    def __init__(
        self,
        sgdw: BaseSGDW[_DistributionT, _PosWgt],
        plot_every=12,
        n_cols=12,
        n_rows=1,
        factor=1.5,
        cmap="binary"
    ):
        super().__init__(sgdw, plot_every, n_cols, n_rows, factor, cmap)
        if n_rows * n_cols > plot_every:
            msg = (f"'plot_every' should not be less than n_rows * n_cols."
                   f" Currently: {plot_every = } < {n_rows * n_cols = }")
            _log.error(msg)
            raise ValueError(msg)
        self.pos_wgt_hist = True
        self.pos_wgt_samp_hist = True

    def plot(self, init: int = None):
        create_distr = self.sgdw.create_distribution
        max_imgs = self.n_rows * self.n_cols
        max_k = self.sgdw.iter_params.k
        init = max_k - max_imgs + 1 if init is None else init
        if init < 0:
            msg = f"init should be greater than 0. Currently: {init = }"
            _log.error(msg)
            raise ValueError(msg)
        if init > max_k - max_imgs + 1:
            msg = f"init should be less than {max_k - max_imgs + 1}. Currently: {init = }"
            _log.error(msg)
            raise ValueError(msg)

        row, col = self.n_rows * 2, self.n_cols

        fig, ax = plt.subplots(
            row, col, figsize=(col * self.factor, row * self.factor),
            subplot_kw={"xticks": [], "yticks": []}
        )

        fig.suptitle("SGDW")

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                k = init + j + i * self.n_cols
                ax0, ax1 = ax[i * 2, j], ax[i * 2 + 1, j]

                # Label the y-axis
                if j == 0:
                    ax0.set_ylabel("Sample")
                    ax1.set_ylabel("Step")

                # Plot the sample
                fig_sample: DistributionDraw = create_distr(self.hist.pos_wgt_samp[k][0])
                ax0.imshow(fig_sample.image, cmap=self.cmap)
                ax0.set_title(f"$k={k}$")

                # Plot the step
                fig_step: DistributionDraw = create_distr(self.hist.pos_wgt[k])
                ax1.imshow(fig_step.image, cmap=self.cmap)

        plt.tight_layout(pad=0.3)

        plt.show()

        return fig, ax
