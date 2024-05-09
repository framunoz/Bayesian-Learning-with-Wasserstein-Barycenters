"""
Module that contains utility functions.
"""
import collections as c
import functools
import time
import typing as t
import warnings

import ipyplot
import numpy as np
import PIL.Image
import torch

import bwb._logging as logging
from bwb.utils.protocols import (device_t, DiscreteDistribSamplerP, DrawP,
                                 seed_t)

__all__ = [
    "set_generator",
    "timeit_to_total_time",
    "freq_labels_dist_sampler",
    "normalised_samples_ordered_dict",
]

_log = logging.get_logger(__name__)


def timeit_to_total_time(method):
    """Function that records the total time it takes to execute a
    method, and stores it in the ``total_time`` attribute of the class
    instance."""

    # noinspection PyMissingOrEmptyDocstring
    @functools.wraps(method)
    def timeit_wrapper(*args, **kwargs):
        tic = time.perf_counter()
        result = method(*args, **kwargs)
        toc = time.perf_counter()
        args[0].total_time += toc - tic
        return result

    return timeit_wrapper


def set_generator(
    seed: seed_t = None,
    device: device_t = "cpu"
) -> torch.Generator:
    """
    Set the generator for the random number generator.

    :param seed: The seed for the random number generator.
    :param device: The device to set the generator.
    :return: The generator.
    """
    if isinstance(seed, torch.Generator):
        if not seed.device == device:
            msg = "The device of the generator must be the same as the device."
            _log.error(msg)
            raise ValueError(msg)
        return seed

    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
        return gen
    gen.manual_seed(seed)
    return gen


def freq_labels_dist_sampler(
    dist_sampler: DiscreteDistribSamplerP
) -> t.Sequence[str]:
    """
    Function that returns the most common labels in the dist_sampler.

    :param dist_sampler: The distribution sampler to analyse.
    :return: A list of strings with the most common labels in the
        dist_sampler.
    """
    if not isinstance(dist_sampler, DiscreteDistribSamplerP):
        msg = ("The dist_sampler must be an instance of "
               "PDiscreteDistribSampler.")
        _log.error(msg)
        raise TypeError(msg)
    return [
        f"id: {id_},\nfreq: {freq}"
        for id_, freq in dist_sampler.samples_counter.most_common()
    ]


def normalised_samples_ordered_dict(posterior: DiscreteDistribSamplerP):
    """
    Function that returns an ordered dictionary with the normalised
    samples in the posterior instance.

    :param posterior: The posterior instance to analyse.
    :return: An ordered dictionary with the normalised samples in the
        MCMC instance.
    """
    counter = posterior.samples_counter
    return c.OrderedDict([(k, v / counter.total())
                          for k, v in counter.most_common()])


# ================= DEPRECATED FUNCTIONS =================
def plot_list_of_images(list_of_images: t.Sequence[PIL.Image.Image], **kwargs):
    """
    Function that plots a list of images.

    :param list_of_images: The list of images to draw.
    :param kwargs: Optional arguments to pass to the ipyplot.plot_images
        function. For further information, please see the documentation of
        that function.
    """
    msg = ("The plot_list_of_images function is deprecated and will be "
           "removed in the future. Use plotters.plot_list_of_images instead.")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Set values by default
    kwargs.setdefault("max_images", 36)
    kwargs.setdefault("img_width", 75)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipyplot.plot_images(list_of_images, **kwargs)


def plot_list_of_draws(list_of_draws: t.Sequence[DrawP], **kwargs):
    """
    Function that plots a list of DistributionDraws instances.

    :param list_of_draws: The list of distributions to draw.
    :param kwargs: Optional arguments to pass to the ipyplot.plot_images
        function. For further information, please see the documentation of
        that function.
    """
    msg = ("The plot_list_of_draws function is deprecated and will be "
           "removed in the future. Use plotters.plot_list_of_draws instead.")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return plot_list_of_images([draw.image for draw in list_of_draws],
                               **kwargs)


def likelihood_ordered_dict(posterior):
    """
    Function that returns an ordered dictionary with the likelihoods of
    the samples in the posterior.

    :param posterior: The posterior to analyse.
    :return: An ordered dictionary with the likelihoods of the samples
        in the posterior.
    """
    msg = ("The likelihood_ordered_dict function is deprecated and will "
           "be removed in the future. ")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    like_cache = posterior.likelihood_cache
    posterior_probs = like_cache / np.sum(like_cache)
    likelihood_dct = c.OrderedDict({i: prob
                                    for i, prob in enumerate(posterior_probs)})

    for key, _ in sorted(likelihood_dct.items(), key=lambda item: -item[1]):
        likelihood_dct.move_to_end(key)

    return likelihood_dct


def normalised_steps_ordered_dict(mcmc):
    """
    Function that returns an ordered dictionary with the normalised
    steps in the MCMC instance.

    :param mcmc: The MCMC instance to analyse.
    :return: An ordered dictionary with the normalised steps in the
        MCMC instance.
    """
    msg = ("The normalised_steps_ordered_dict function is deprecated "
           "and will be removed in "
           "the future.")
    _log.warning(msg)
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    counter = mcmc.steps_counter
    return c.OrderedDict([(k, v / counter.total())
                          for k, v in counter.most_common()])
