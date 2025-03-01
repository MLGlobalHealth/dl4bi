from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax, softplus
from jax.scipy.stats import norm
from optax.losses import safe_softmax_cross_entropy

from ..core.metrics import mvn_logpdf

# TODO(danj):
# Support Binomial and Poisson


@dataclass(frozen=True)
class ModelOutput(Mapping):
    """A generic model output class."""

    def __getitem__(self, key):
        return asdict(self)[key]

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        return iter(asdict(self))

    def __len__(self):
        return len(asdict(self))


@dataclass(frozen=True)
class DistributionOutput(ModelOutput, ABC):
    @abstractmethod
    def nll(self, *args, **kwargs):
        raise NotImplementedError()


@dataclass(frozen=True)
class DiagonalMVNOutput(DistributionOutput):
    mu: jax.Array
    std: jax.Array

    @classmethod
    def from_conditional_np(cls, params: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(params, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        return DiagonalMVNOutput(mu, std)

    @classmethod
    def from_latent_np(cls, params: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(params, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        # average over latent n_z samples
        return DiagonalMVNOutput(mu.mean(axis=1), std.mean(axis=1))

    def nll(self, x: jax.Array, **kwargs):
        return -norm.logpdf(x, self.mu, self.std)

    def forward_kl_div(self, p: "DiagonalMVNOutput"):
        return forward_kl_div(p, self)

    def reverse_kl_div(self, p: "DiagonalMVNOutput"):
        return forward_kl_div(self, p)


@jit
def forward_kl_div(p: DiagonalMVNOutput, q: DiagonalMVNOutput):
    # KL divergence and NLL assume diagonal covariance, i.e. pointwise.
    # Wikipedia's formulas for MVN KL-div: https://tinyurl.com/wiki-kl-div
    # Tensorflow's diagonal MVN KL-div impl (used here): https://tinyurl.com/diag-kl-div
    # KL( z_dist_test (p) || z_dist_ctx (q) ) =
    diff_log_scale = jnp.log(p.std) - jnp.log(q.std)
    return (
        0.5 * ((p.mu - q.mu) / q.std) ** 2
        + 0.5 * jnp.expm1(2 * diff_log_scale)
        - diff_log_scale
    ).sum(axis=-1)


@dataclass(frozen=True)
class TrilMVNOutput(DistributionOutput):
    mu: jax.Array
    L: jax.Array

    def nll(self, x: jax.Array, **kwargs):
        B = x.shape[0]
        x, mu = x.reshape(B, -1), self.mu.reshape(B, -1)
        return -mvn_logpdf(x, mu, self.L, is_tril=True)


@dataclass(frozen=True)
class MultinomialOutput(DistributionOutput):
    logits: jax.Array

    @property
    def p(self):
        return softmax(self.logits, axis=-1)

    @property
    def std(self):
        return jnp.sqrt(self.p * (1 - self.p))

    @classmethod
    def from_conditional_np(cls, logits: jax.Array, **kwargs):
        return MultinomialOutput(logits)

    @classmethod
    def from_latent_np(cls, logits: jax.Array, **kwargs):
        # average over n_z latent samples
        return MultinomialOutput(logits.mean(axis=1))

    def nll(self, x: jax.Array, **kwargs):
        return safe_softmax_cross_entropy(self.logits, x)


@dataclass(frozen=True)
class LatentOutput(ModelOutput):
    dist: DistributionOutput
