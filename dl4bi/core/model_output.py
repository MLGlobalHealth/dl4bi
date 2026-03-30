"""Structured model outputs and distribution-specific losses."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.nn import sigmoid, softmax, softplus
from jax.scipy import stats
from jax.scipy.special import gammaln, logsumexp
from optax.losses import safe_softmax_cross_entropy, squared_error
from scipy.stats import poisson


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
    """Abstract base class for probabilistic model outputs."""

    @abstractmethod
    def nll(self, *args, **kwargs):
        """Return the negative log-likelihood under this output distribution."""
        raise NotImplementedError()


@dataclass(frozen=True)
class DiagonalMVNOutput(DistributionOutput):
    """Output wrapper for diagonal multivariate normal predictions."""

    mu: jax.Array
    std: jax.Array

    @classmethod
    def from_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        """Build a diagonal MVN output from raw network activations."""
        mu, std = jnp.split(act, 2, axis=-1)
        std = min_std + (1 - min_std) * softplus(std)
        return DiagonalMVNOutput(mu, std)

    @classmethod
    def from_latent_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        """Build a diagonal MVN output from latent-sample activations."""
        mu, std = jnp.split(act, 2, axis=-1)
        mu, std = mu.mean(axis=1), std.mean(axis=1)  # average over latent n_z samples
        std = min_std + (1 - min_std) * softplus(std)
        return DiagonalMVNOutput(mu, std)

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute the masked negative log-likelihood."""
        return -stats.norm.logpdf(x, self.mu, self.std).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute standard regression metrics for the predictive distribution."""
        hdi_prob = kwargs.get("hdi_prob", 0.95)
        z_score = jnp.abs(stats.norm.ppf((1 - hdi_prob) / 2))
        rmse = jnp.sqrt(jnp.square(x - self.mu).mean(where=mask))
        mae = jnp.abs(x - self.mu).mean(where=mask)
        f_lower, f_upper = self.mu - z_score * self.std, self.mu + z_score * self.std
        cvg = ((x >= f_lower) & (x <= f_upper)).mean(where=mask)
        return {"NLL": self.nll(x, mask), "RMSE": rmse, "MAE": mae, "Coverage": cvg}

    def forward_kl_div(self, p: "DiagonalMVNOutput"):
        """Compute ``KL(p || self)`` averaged over the event dimension."""
        return forward_kl_div(p, self).mean()

    def reverse_kl_div(self, p: "DiagonalMVNOutput"):
        """Compute ``KL(self || p)`` averaged over the event dimension."""
        return forward_kl_div(self, p).mean()


@jit
def forward_kl_div(p: DiagonalMVNOutput, q: DiagonalMVNOutput):
    """Compute the forward KL divergence ``KL(p || q)`` for diagonal MVNs."""
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
    """Output wrapper for categorical or multinomial logits."""

    logits: jax.Array

    @property
    def p(self):
        """Return class probabilities."""
        return softmax(self.logits, axis=-1)

    @property
    def std(self):
        """Return the per-class Bernoulli standard deviation."""
        return jnp.sqrt(self.p * (1 - self.p))

    @classmethod
    def from_activations(cls, act: jax.Array, **kwargs):
        """Build a multinomial output from raw logits."""
        return MultinomialOutput(act)

    @classmethod
    def from_latent_activations(cls, act: jax.Array, **kwargs):
        """Build a multinomial output by averaging latent logits."""
        # average over n_z latent samples
        return MultinomialOutput(act.mean(axis=1))

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute the masked categorical negative log-likelihood."""
        mask = None if mask is None else mask[..., 0]
        return safe_softmax_cross_entropy(self.logits, x).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute summary metrics for categorical predictions."""
        return {"NLL": self.nll(x, mask)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    MultinomialOutput,
    lambda d: ((d.logits,), None),
    lambda _aux, children: MultinomialOutput(*children),
)


@dataclass(frozen=True)
class BetaOutput(DistributionOutput):
    """Output wrapper for Beta-distributed targets on ``[0, 1]``."""

    alpha: jax.Array
    beta: jax.Array

    @classmethod
    def from_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        """Build a Beta output from raw network activations."""
        alpha, beta = jnp.split(act, 2, axis=-1)
        return BetaOutput(softplus(alpha), softplus(beta))

    @classmethod
    def from_latent_activations(cls, act: jax.Array, min_std: float = 0.0, **kwargs):
        """Build a Beta output by averaging latent activations."""
        alpha, beta = jnp.split(act, 2, axis=-1)
        alpha, beta = (alpha.mean(axis=1), beta.mean(axis=1))  # average latent samples
        return BetaOutput(softplus(alpha), softplus(beta))

    @property
    def p(self):
        """Return the predictive mean."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self):
        """Return the predictive standard deviation."""
        a, b = self.alpha, self.beta
        return jnp.sqrt(a * b / (jnp.square(a + b) * (a + b + 1)))

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute the masked Beta negative log-likelihood."""
        mask = None if mask is None else mask[..., 0]
        return -stats.beta.logpdf(x, self.alpha, self.beta).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute summary metrics for Beta predictions."""
        return {"NLL": self.nll(x, mask)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    BetaOutput,
    lambda d: ((d.alpha, d.beta), None),
    lambda _aux, children: BetaOutput(*children),
)


@dataclass(frozen=True)
class PoissonOutput(DistributionOutput):
    """Output wrapper for Poisson-distributed count targets."""

    lam: jax.Array

    @classmethod
    def from_activations(cls, act: jax.Array, **kwargs):
        """Build a Poisson output from raw network activations."""
        return PoissonOutput(softplus(act))

    @classmethod
    def from_latent_activations(cls, act: jax.Array, **kwargs):
        """Build a Poisson output by averaging latent activations."""
        act = act.mean(axis=1)  # average latent samples
        return PoissonOutput(softplus(act))

    @property
    def mu(self):
        """Return the predictive mean."""
        return self.lam

    @property
    def var(self):
        """Return the predictive variance."""
        return self.lam

    def ci(self, lower: float = 0.05, upper: float = 0.95):
        """Return lower and upper Poisson quantiles."""
        return poisson.ppf(lower, self.lam), poisson.ppf(upper, self.lam)

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute the masked Poisson negative log-likelihood."""
        return -stats.poisson.logpmf(x, self.lam).mean(where=mask)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute summary metrics for Poisson predictions."""
        rmse = jnp.sqrt(jnp.square(x - self.lam).mean(where=mask))
        mae = jnp.abs(x - self.lam).mean(where=mask)
        return {"NLL": self.nll(x, mask), "RMSE": rmse, "MAE": mae}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    PoissonOutput,
    lambda d: ((d.lam,), None),
    lambda _aux, children: PoissonOutput(*children),
)


@dataclass(frozen=True)
class ZeroInflatedPoissonOutput(DistributionOutput):
    """Output wrapper for zero-inflated Poisson count targets."""

    pi: jax.Array
    lam: jax.Array

    @classmethod
    def from_activations(cls, act: jax.Array, **kwargs):
        """Build a zero-inflated Poisson output from raw activations."""
        log_pi, log_lam = jnp.split(act, 2, axis=-1)
        return ZeroInflatedPoissonOutput(sigmoid(log_pi), softplus(log_lam))

    @classmethod
    def from_latent_activations(cls, act: jax.Array, **kwargs):
        """Build a zero-inflated Poisson output by averaging latent activations."""
        act = act.mean(axis=1)  # average latent samples
        return ZeroInflatedPoissonOutput.from_activations(act)

    @property
    def mu(self):
        """Return the predictive mean."""
        return (1 - self.pi) * self.lam

    @property
    def var(self):
        """Return the predictive variance."""
        return (1 - self.pi) * self.lam * (1 + self.pi * self.lam)

    def ci(self, lower: float = 0.05, upper: float = 0.95, max_k: int = 10000):
        """Return lower and upper zero-inflated Poisson quantiles."""
        vec_q_lower = np.vectorize(lambda pi, lam: _zip_quantile(lower, pi, lam, max_k))
        vec_q_upper = np.vectorize(lambda pi, lam: _zip_quantile(upper, pi, lam, max_k))
        return vec_q_lower(self.pi, self.lam), vec_q_upper(self.pi, self.lam)

    def nll(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute the zero-inflated Poisson negative log-likelihood."""
        pi, lam = self.pi, self.lam
        log_p_0 = jnp.logaddexp(jnp.log(pi), jnp.log1p(-pi) - lam)
        log_p_gt_0 = jnp.log1p(-pi) + x * jnp.log(lam) - lam - gammaln(x + 1)
        log_p = jnp.where(x == 0, log_p_0, log_p_gt_0)
        return -jnp.mean(log_p)

    def metrics(self, x: jax.Array, mask: Optional[jax.Array] = None, **kwargs):
        """Compute summary metrics for zero-inflated Poisson predictions."""
        return {"NLL": self.nll(x, mask)}


def _zip_quantile(alpha: float, pi: float, lam: int, max_k: int = 10000):
    """Approximate a zero-inflated Poisson quantile by cumulative search."""
    if alpha <= pi + (1 - pi) * np.exp(-lam):
        return 0.0
    for k in range(1, max_k + 1):
        if _zip_cdf(k, pi, lam) >= alpha:
            return k
    raise ValueError(f"quantile not found up to k={max_k}")


def _zip_cdf(k: int, pi: float, lam: int):
    """Evaluate the zero-inflated Poisson cumulative distribution function."""
    c0 = pi + (1 - pi) * poisson.pmf(0, lam)
    if k < 0:
        return 0.0
    if k == 0:
        return c0
    return c0 + (1 - pi) * (poisson.cdf(k, lam) - poisson.pmf(0, lam))


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    ZeroInflatedPoissonOutput,
    lambda d: ((d.pi, d.lam), None),
    lambda _aux, children: ZeroInflatedPoissonOutput(*children),
)


@dataclass(frozen=True)
class MDNOutput(DistributionOutput):
    """Output wrapper for mixture density network predictions."""

    pi_logits: jax.Array  # [B, K]
    mu: jax.Array  # [B, K]
    std: jax.Array  # [B, K]

    @property
    def pi(self):
        """Return mixture weights."""
        return nn.softmax(self.pi_logits, axis=-1)

    @classmethod
    def from_activations(cls, act: jax.Array, min_std: float = 1e-5, **kwargs):
        """Build a mixture density output from raw activations."""
        pi_logits, mu, std = jnp.split(act, 3, axis=-1)
        pi = nn.softmax(pi_logits, axis=-1)
        std = min_std + (1 - min_std) * nn.softplus(std)
        return MDNOutput(pi, mu, std)

    def nll(self, x: jax.Array, **kwargs):
        """Compute the mixture negative log-likelihood."""
        x = x[None, :] if x.ndim == 1 else x  # x: [B, 1]
        ll = stats.norm.logpdf(x, self.mu, self.std)
        ll = logsumexp(self.pi_logits + ll, axis=-1)
        return -ll.mean()

    def metrics(self, x: jax.Array, **kwargs):
        """Compute summary metrics for mixture density predictions."""
        return {"NLL": self.nll(x)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    MDNOutput,
    lambda d: ((d.pi_logits, d.mu, d.std), None),
    lambda _aux, children: MDNOutput(*children),
)


@dataclass(frozen=True)
class VAEOutput(DistributionOutput):
    """Output wrapper for variational autoencoder reconstructions."""

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
        """Build a VAE output from reconstruction and latent statistics."""
        latent_mu = jnp.atleast_3d(latent_mu)
        latent_std = jnp.atleast_3d(latent_std)
        return VAEOutput(f_hat, DiagonalMVNOutput(latent_mu, latent_std))

    def nll(self, f: jax.Array, var: Optional[float] = None, **kwargs):
        """Compute the reconstruction negative log-likelihood."""
        std = jnp.sqrt(var) if var is not None else 1.0
        ll = stats.norm.logpdf(self.f_hat.squeeze(), f.squeeze(), std)
        return -ll.mean()

    def kl_normal_dist(self, **kwargs):
        """Compute the KL divergence to a unit normal prior."""
        normal_dist = DiagonalMVNOutput(jnp.array(0.0), jnp.array(1.0))
        if self.encoder_outputs is not None:
            return self.encoder_outputs.reverse_kl_div(normal_dist)
        return 0.0

    def mse(self, f: jax.Array):
        """Compute the mean squared reconstruction error."""
        return squared_error(self.f_hat.squeeze(), f.squeeze()).mean()

    def metrics(self, f: jax.Array, var: Optional[float] = None, **kwargs):
        """Compute summary metrics for VAE reconstructions."""
        return {"NLL": self.nll(f, var), "MSE": self.mse(f)}


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    VAEOutput,
    lambda d: ((d.f_hat, d.encoder_outputs), None),
    lambda _aux, children: VAEOutput(*children),
)
