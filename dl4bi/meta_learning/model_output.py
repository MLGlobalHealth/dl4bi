from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax, softplus
from jax.scipy.stats import norm
from optax.losses import safe_softmax_cross_entropy

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

    def nll(self, x: jax.Array, mask: Optional[jax.Array], **kwargs):
        if mask is not None:
            mask = mask[..., None]
        return -norm.logpdf(x, self.mu, self.std).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array], **kwargs):
        if mask is not None:
            mask = mask[..., None]
        hdi_prob = kwargs.get("hdi_prob", 0.95)
        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        rmse = jnp.sqrt(jnp.square(x - self.mu).mean(where=mask))
        mae = jnp.abs(x - self.mu).mean(where=mask)
        f_lower, f_upper = self.mu - z_score * self.std, self.mu + z_score * self.std
        cvg = (x >= f_lower) & (x <= f_upper)
        return {"NLL": self.nll(x, mask), "RMSE": rmse, "MAE": mae, "Coverage": cvg}

    def forward_kl_div(self, p: "DiagonalMVNOutput"):
        return forward_kl_div(p, self).mean()

    def reverse_kl_div(self, p: "DiagonalMVNOutput"):
        return forward_kl_div(self, p).mean()


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


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    DiagonalMVNOutput,
    lambda d: ((d.mu, d.std), None),
    lambda _aux, children: DiagonalMVNOutput(*children),
)


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

    def nll(self, x: jax.Array, mask: Optional[jax.Array], **kwargs):
        return safe_softmax_cross_entropy(self.logits, x).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array]):
        return {"NLL": self.nll(x, mask)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    MultinomialOutput,
    lambda d: ((d.logits,), None),
    lambda _aux, children: MultinomialOutput(*children),
)
