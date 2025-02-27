from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax, softplus
from jax.scipy.stats import norm
from optax.losses import safe_softmax_cross_entropy

# TODO(danj):
# 1. Support bootstrapped data
# 2. Support Binomial and Poisson


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
class GaussianOutput(DistributionOutput):
    mu: jax.Array
    std: jax.Array

    @classmethod
    def from_conditional(cls, dist: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(dist, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        return GaussianOutput(mu, std)

    @classmethod
    def from_latent(cls, dist: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(dist, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        return mu.mean(axis=1), std.mean(axis=1)  # average over n_z latent samples

    def nll(self, x: jax.Array, **kwargs):
        return -norm.logpdf(x, self.mu, self.std)


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
    def from_conditional(cls, logits: jax.Array, **kwargs):
        return MultinomialOutput(logits)

    @classmethod
    def from_latent(cls, logits: jax.Array, **kwargs):
        # average over n_z latent samples
        return MultinomialOutput(logits.mean(axis=1))

    def nll(self, x: jax.Array, **kwargs):
        return safe_softmax_cross_entropy(self.logits, x)


@partial(jit, static_argnames=("min_std",))
def diagonal_mvn(f_dist: jax.Array, min_std: float = 0.0):
    f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
    f_std = min_std + (1 - min_std) * softplus(f_std)
    return f_mu, f_std


@partial(jit, static_argnames=("min_std",))
def latent_diagonal_mvn(f_dist: jax.Array, min_std: float = 0.0):
    f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
    f_std = min_std + (1 - min_std) * softplus(f_std)
    return f_mu.mean(axis=1), f_std.mean(axis=1)  # average over n_z latent samples


@jit
def identity(output: jax.Array):
    return output


@jit
def latent_logits(logits: jax.Array):
    return logits.mean(axis=1)  # average over n_z latent samples


@jit
def pointwise_multinomial(f_dist: jax.Array):
    p = softmax(f_dist, axis=-1)
    return p, p * (1 - p)
