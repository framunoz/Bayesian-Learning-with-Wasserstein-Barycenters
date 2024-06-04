"""
Module that contains functions to plot images and histograms.
"""
import typing as t
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import seaborn as sns
from matplotlib import pyplot as plt

import bwb.logging_ as logging

__all__ = [
    "plot_image",
    "plot_draw",
    "plot_list_of_images",
    "plot_list_of_draws",
    "plot_histogram_from_points",
]

_log = logging.get_logger(__name__)

_CMAP_DEFAULT = "binary_r"


# noinspection PyMissingOrEmptyDocstring,PyPropertyDefinition
class DistributionP(t.Protocol):
    @property
    def image(self) -> PIL.Image.Image:
        ...


def save_fig(
    fig: plt.Figure,
    name: str,
    path: Path | str,
    formats: t.Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
    **kwargs,
) -> None:
    """
    Function that saves a figure.

    :param fig: The figure to save.
    :type fig: plt.Figure
    :param name: The name of the file.
    :type name: str
    :param path: The path of the file.
    :type path: Path | str
    :param formats: The formats of the image.
    :type formats: list[str]
    :param dpi: The resolution of the image.
    :type dpi: int
    :param kwargs: Optional arguments to pass to the
        `matplotlib.pyplot.savefig
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html>`_
        function. For further information, please see the documentation
        of that function.
    """
    path_to_save = Path(path) / name
    for fmt in formats:
        path_to_save_with_format = path_to_save.with_suffix(f".{fmt}")
        fig.savefig(path_to_save_with_format, dpi=dpi, **kwargs)


def save_fig_with_path(
    path: Path | str,
    formats: t.Sequence[str] = ("pdf", "png"),
    **kwargs,
):
    """
    Function that returns a function that saves a figure with a given
    path.

    :param path: The path of the file.
    :type path: Path | str
    :param formats: The formats of the image. Default is ("pdf", "png").
    :type formats: Sequence[str]
    :return: The function that saves the figure.
    """

    def _save_fig(fig: plt.Figure, name: str, **_kwargs) -> None:
        # Update the kwargs
        _kwargs.update(kwargs)
        return save_fig(fig, name, path, formats, **_kwargs)

    _save_fig.__doc__ = save_fig.__doc__
    return _save_fig


def plot_image(
    image: PIL.Image.Image,
    title: str = "Image",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Function that plots an image.

    :param image: The image to draw.
    :type image: PIL.Image.Image
    :param title: The title of the plot.
    :type title: str
    :param kwargs: Optional arguments to pass to the
        `matplotlib.pyplot.imshow
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
        function. For further information, please see the documentation
        of that function.
    :return: The figure and the axes of the plot.
    """
    kwargs.setdefault("cmap", _CMAP_DEFAULT)

    fig, ax = plt.subplots()
    ax.imshow(image, **kwargs)
    ax.set_title(title)
    ax.axis("off")
    plt.show()

    return fig, ax


def plot_draw(
    draw: DistributionP,
    title: str = "Draw",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Function that plots a DistributionDraw instance.

    :param draw: The DistributionDraw instance to draw.
    :type draw: DistributionDraw
    :param title: The title of the plot.
    :type title: str
    :param kwargs: Optional arguments to pass to the
        `matplotlib.pyplot.imshow
        <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_
        function. For further information, please see the documentation
        of that function.
    :return: The figure and the axes of the plot.
    """
    return plot_image(image=draw.image, title=title, **kwargs)


def plot_list_of_images(
    list_of_images: t.Sequence[PIL.Image.Image],
    n_rows: int = 3, n_cols: int = 12, factor: float = 1.5,
    title=None, cmap: str = _CMAP_DEFAULT,
    labels: t.Optional[t.Sequence[str]] = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Function that plots a list of images.

    :param list_of_images: The list of images to draw.
    :type list_of_images: list[PIL.Image.Image]
    :param n_rows: The number of rows of the plot.
    :type n_rows: int
    :param n_cols: The number of columns of the plot.
    :type n_cols: int
    :param factor: The factor to multiply the number of rows by the
        number of columns.
    :type factor: float
    :param title: The title of the plot.
    :type title: str
    :param labels: The labels of the images.
    :type labels: list[str]
    :param cmap: The colormap of the plot.
    :type cmap: str
    :return: The figure and the axes of the plot.
    """

    n_images = len(list_of_images)
    if n_images < n_rows * n_cols:
        msg = ("The number of images is less than the number of rows "
               "times the number of columns.")
        _log.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    n_images = min(n_images, n_rows * n_cols)

    if labels is None:
        labels = [f"{i}" for i in range(n_images)]

    if len(labels) < n_images:
        msg = "The number of labels is different from the number of images."
        _log.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * factor, n_rows * factor),
        subplot_kw={"xticks": [], "yticks": []}
    )  # type: plt.Figure, np.ndarray[plt.Axes]

    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axs.flat):  # type: int, plt.Axes
        if i < n_images:
            ax.imshow(list_of_images[i], cmap=cmap)
            ax.set_xlabel(labels[i])
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

    return fig, axs


def plot_list_of_draws(
    list_of_draws: t.Sequence[DistributionP],
    n_rows: int = 4, n_cols=12, factor: float = 1.5,
    title=None, cmap: str = _CMAP_DEFAULT,
    labels: t.Optional[t.Sequence[str]] = None,
) -> tuple[plt.Figure, np.ndarray[plt.Axes]]:
    """
    Function that plots a list of DistributionDraws instances.

    :param list_of_draws: The list of DistributionDraws instances to
        draw.
    :type list_of_draws: list[DistributionDraws]
    :param n_rows: The number of rows of the plot.
    :type n_rows: int
    :param n_cols: The number of columns of the plot.
    :type n_cols: int
    :param factor: The factor to multiply the number of rows by the
        number of columns.
    :type factor: float
    :param title: The title of the plot.
    :type title: str
    :param labels: The labels of the images.
    :type labels: list[str]
    :param cmap: The colormap of the plot.
    :type cmap: str

    """

    list_of_images: list[PIL.Image.Image] = [
        draw.image for draw in list_of_draws[:n_rows * n_cols]
    ]
    return plot_list_of_images(
        list_of_images=list_of_images,
        n_rows=n_rows, n_cols=n_cols, factor=factor,
        title=title, cmap=cmap, labels=labels
    )


def plot_histogram_from_points(
    data: list[tuple[int, int]],
    shape: tuple[int, int] = (28, 28),
    rotate: bool = True,
    title: str = "Histogram of the distribution generated by a drawing",
    xlabel: str = None,
    ylabel: str = None,
    histplot_kwargs: t.Optional[dict] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Function that plots a histogram from a list of points.

    :param data: The list of points to draw.
    :type data: list[tuple[int, int]]
    :param shape: The shape of the image.
    :type shape: tuple[int, int]
    :param rotate: If True, the image is rotated 90 degrees.
    :param title: The title of the plot.
    :param xlabel: The label of the x-axis.
    :param ylabel: The label of the y-axis.
    :param histplot_kwargs: Optional arguments to pass to the
        `seaborn.histplot
        <https://seaborn.pydata.org/generated/seaborn.histplot.html>`_
        function. For further information, please see the documentation
        of that function.
    :return: The return of the ``seaborn.histplot`` function.
    """
    # Instance the kwargs of the histplot and set default values.
    histplot_kwargs = dict() if histplot_kwargs is None else histplot_kwargs
    histplot_kwargs.setdefault("bins", 100)
    histplot_kwargs.setdefault("cbar", True)
    histplot_kwargs.setdefault("binrange", ((0, shape[0]), (0, shape[1])))

    xlabel_ = "Y-Axis" if xlabel is None else xlabel
    ylabel_ = "X-Axis" if ylabel is None else ylabel

    if rotate:
        data = [(point[1], shape[1] - 1 - point[0]) for point in data]
        xlabel_ = "X-Axis" if xlabel is None else xlabel
        ylabel_ = "Y-Axis" if ylabel is None else ylabel

    df = pd.DataFrame(data)
    histplot_return = sns.histplot(data=df, x=0, y=1,
                                   **histplot_kwargs)  # type: plt.Axes
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    plt.title(title)

    return histplot_return.figure, histplot_return
