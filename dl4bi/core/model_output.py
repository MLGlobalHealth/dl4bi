from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax, softplus
from jax.scipy import stats
from jax.scipy.special import logsumexp
from optax.losses import safe_softmax_cross_entropy, squared_error


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
    def from_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(act, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        return DiagonalMVNOutput(mu, std)

    @classmethod
    def from_latent_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        mu, std = jnp.split(act, 2, axis=-1)
        mu, std = mu.mean(axis=1), std.mean(axis=1)  # average over latent n_z samples
        std = min_std + (1 - min_std) * softplus(std)
        return DiagonalMVNOutput(mu, std)

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        return -stats.norm.logpdf(x, self.mu, self.std).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        hdi_prob = kwargs.get("hdi_prob", 0.95)
        z_score = jnp.abs(stats.norm.ppf((1 - hdi_prob) / 2))
        rmse = jnp.sqrt(jnp.square(x - self.mu).mean(where=mask))
        mae = jnp.abs(x - self.mu).mean(where=mask)
        f_lower, f_upper = self.mu - z_score * self.std, self.mu + z_score * self.std
        cvg = ((x >= f_lower) & (x <= f_upper)).mean(where=mask)
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
    def from_activations(cls, act: jax.Array, **kwargs):
        return MultinomialOutput(act)

    @classmethod
    def from_latent_activations(cls, act: jax.Array, **kwargs):
        # average over n_z latent samples
        return MultinomialOutput(act.mean(axis=1))

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        mask = None if mask is None else mask[..., 0]
        return safe_softmax_cross_entropy(self.logits, x).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        return {"NLL": self.nll(x, mask)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    MultinomialOutput,
    lambda d: ((d.logits,), None),
    lambda _aux, children: MultinomialOutput(*children),
)


@dataclass(frozen=True)
class BetaOutput(DistributionOutput):
    alpha: jax.Array
    beta: jax.Array

    @classmethod
    def from_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        alpha, beta = jnp.split(act, 2, axis=-1)
        return BetaOutput(softplus(alpha), softplus(beta))

    @classmethod
    def from_latent_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        alpha, beta = jnp.split(act, 2, axis=-1)
        alpha, beta = (alpha.mean(axis=1), beta.mean(axis=1))  # average latent samples
        return BetaOutput(softplus(alpha), softplus(beta))

    @property
    def p(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self):
        a, b = self.alpha, self.beta
        return jnp.sqrt(a * b / (jnp.square(a + b) * (a + b + 1)))

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        mask = None if mask is None else mask[..., 0]
        return -stats.beta.logpdf(x, self.alpha, self.beta).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        return {"NLL": self.nll(x, mask)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    BetaOutput,
    lambda d: ((d.alpha, d.beta), None),
    lambda _aux, children: BetaOutput(*children),
)


@dataclass(frozen=True)
class MDNOutput(DistributionOutput):
    pi_logits: jax.Array  # [B, K]
    mu: jax.Array  # [B, K]
    std: jax.Array  # [B, K]

    @property
    def pi(self):
        return nn.softmax(self.pi_logits, axis=-1)

    @classmethod
    def from_activations(cls, act: jax.Array, min_std: float = 1e-5, **kwargs):
        pi_logits, mu, std = jnp.split(act, 3, axis=-1)
        pi = nn.softmax(pi_logits, axis=-1)
        std = min_std + (1 - min_std) * nn.softplus(std)
        return MDNOutput(pi, mu, std)

    def nll(self, x: jax.Array, **kwargs):
        x = x[None, :] if x.ndim == 1 else x  # x: [B, 1]
        ll = stats.norm.logpdf(x, self.mu, self.std)
        ll = logsumexp(self.pi_logits + ll, axis=-1)
        return -ll.mean()

    def metrics(self, x: jax.Array, **kwargs):
        return {"NLL": self.nll(x)}


@dataclass(frozen=True)
class VAEOutput(DistributionOutput):
    f_hat: jax.Array
    encoder_outputs: Optional[DiagonalMVNOutput] = None

    @classmethod
    def from_raw_output(
        cls,
        f_hat: jax.Array,
        latent_mu: jax.Array,
        latent_std: jax.Array,
        **kwargs,
    ):
        latent_mu = jnp.atleast_3d(latent_mu)
        latent_std = jnp.atleast_3d(latent_std)
        return VAEOutput(f_hat, DiagonalMVNOutput(latent_mu, latent_std))

    def nll(self, f: jax.Array, var: Optional[float] = None, **kwargs):
        std = jnp.sqrt(var) if var is not None else 1.0
        ll = stats.norm.logpdf(self.f_hat.squeeze(), f.squeeze(), std)
        return -ll.mean()

    def kl_normal_dist(self, **kwargs):
        normal_dist = DiagonalMVNOutput(jnp.array(0.0), jnp.array(1.0))
        if self.encoder_outputs is not None:
            return self.encoder_outputs.reverse_kl_div(normal_dist)
        return 0.0

    def mse(self, f: jax.Array):
        return squared_error(self.f_hat.squeeze(), f.squeeze()).mean()

    def metrics(self, f: jax.Array, var: Optional[float] = None, **kwargs):
        return {"NLL": self.nll(f, var), "MSE": self.mse(f)}
