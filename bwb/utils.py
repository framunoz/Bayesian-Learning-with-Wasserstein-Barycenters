import warnings
from typing import Optional

import PIL.Image
import ipyplot
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from bwb.distributions import DistributionDraw


def plot_histogram_from_points(
        data: list[tuple[int, int]],
        title: str = "Histogram of the distribution generated by a drawing",
        xlabel: str = "Y-Axis",
        ylabel: str = "X-Axis",
        histplot_kwargs: Optional[dict] = None,
):
    # Instance the kwargs of the histplot and set default values.
    histplot_kwargs = dict() if histplot_kwargs is None else histplot_kwargs
    histplot_kwargs.setdefault("bins", 100)
    histplot_kwargs.setdefault("cbar", True)

    df = pd.DataFrame(data)
    histplot_return = sns.histplot(data=df, x=0, y=1, **histplot_kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    return histplot_return


def plot_list_of_draws(list_of_draws: list[DistributionDraw], **kwargs):
    """
    Function that plots a list of DistributionDraws instances.

    Parameters
    ----------
    list_of_draws: list[DistributionDraw]
        The list of distributions to draw.
    kwargs: optional
        Optional arguments to pass to the ipyplot.plot_images function. For further information, please review the
         documentation of that function.

    Returns
    -------

    """
    # Map the list of draws to obtain a list of images
    list_of_images: list[PIL.Image.Image] = [draw.image for draw in list_of_draws]

    # Set values by default
    kwargs.setdefault("max_images", 55)
    kwargs.setdefault("img_width", 75)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipyplot.plot_images(list_of_images, **kwargs)
