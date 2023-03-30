from typing import Sequence

import numpy as np
import torch
from numpy.random import Generator

from bwb.distributions.discrete_distribution import DiscreteDistribution
from bwb.distributions.posterior_distributions.posterior_pi_n import PosteriorPiN


# noinspection PyAttributeOutsideInit
class ExplicitPosteriorPiN(PosteriorPiN):
    r"""Distribution that uses the strategy of calculating all likelihoods by brute force. This class implements
    likelihoods of the form
     .. math::
        \mathcal{L}_n(m) = \prod_{i=1}^{n} \rho_{m}(x_i)

    using the log-likelihood for stability. Finally, to compute the sampling probabilities, for a discrete
    set :math:`\mathcal{M}` of models, using a uniform prior, we have the posterior explicit by
     .. math::
        \Pi_n(m) = \frac{\mathcal{L}_n(m)}{\sum_{\bar m \in \mathcal{M}} \mathcal{L}_n(\bar m)}

     """

    def __init__(
            self,
            log_likelihood_fn=None,
            device=None,
    ):
        super().__init__(log_likelihood_fn, device)

    def fit(
            self,
            data: Sequence[int],
            models: Sequence[DiscreteDistribution],
    ):
        super(ExplicitPosteriorPiN, self).fit(data, models)

        # Compute the log-likelihood of the models as cache
        # Data with shape (1, n_data)
        data = self.data_.reshape(1, -1)

        # logits array with shape (n_models, n_support)
        logits_models = torch.cat([model.logits.reshape(1, -1) for model in self.models_], 0)

        # Take the evaluations of the logits, resulting in a tensor of shape (n_models, n_data)
        evaluations = torch.take_along_dim(logits_models, data, 1)

        # Get the likelihood as cache. The shape is (n_models,)
        likelihood_cache = torch.exp(torch.sum(evaluations, 1))

        # Get the posterior probabilities.
        self.probabilities_: np.ndarray = (likelihood_cache / likelihood_cache.sum()).cpu().numpy()

        return self

    def _draw(self, seed=None):
        rng: Generator = np.random.default_rng(seed)
        i = rng.choice(a=self.models_index_, p=self.probabilities_)
        return self.models_[i], i

    def _rvs(self, size=1, seed=None, **kwargs):
        rng: Generator = np.random.default_rng(seed)
        list_i = list(rng.choice(a=self.models_index_, size=size, p=self.probabilities_))
        return self.models_.take(list_i), list_i

    def draw(self, seed=None, **kwargs):
        super().draw(seed=seed)

    def rvs(self, size=1, seed=None, **kwargs):
        super().rvs(size=size, seed=seed)
