from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit, random
from numpyro import handlers
from numpyro.infer import Predictive
from omegaconf import DictConfig
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from sps.utils import build_grid


def numpyro_model(
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    f: Optional[jax.Array] = None,  # [L, 1]
    mask: Optional[bool] = None,
    jitter: float = 1e-5,
):
    """Generic Spatiotemporal model with fixed and random spatial effects.

    Args:
        s: Array of input locations, `[S]`.
        x: Array of input covariates, `[S, D]`.
        f: Observed function values, `[S, 1]`.
        mask: Masks observations for likelihood, `[S]`.
        jitter: Jitter for kernel covariance to stabilize inversion.
    """

    L, D = x.shape
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s, s, var=1.0, ls=ls) + jitter * jnp.eye(L)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    f_mu_x = x @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(jnp.zeros(L), k))
    f_mu = f_mu_x + f_mu_s
    f_obs_noise = numpyro.sample("f_obs_noise", dist.HalfNormal(0.1))
    with handlers.trace() if mask is None else handlers.mask(mask):
        numpyro.sample("f", dist.Normal(f_mu, f_obs_noise), obs=f)


@partial(jit, static_argnames=("batch_size",))
def numpyro_prior_pred(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    B = batch_size
    prior_pred = Predictive(numpyro_model, num_samples=B)
    samples = prior_pred(rng, x=x, s=s)
    return (
        samples["f"][..., None],
        samples["ls"],
        samples["beta"],
        samples["f_obs_noise"],
    )


def jax_prior_pred(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
    kernel: Callable = rbf,
):
    """A pure JAX spatiotemporalmodel with fixed and random spatial effects.

    Technically this isn't the same model since the GP class samples
    the GP priors once per batch in order to amortize the cost of
    the Cholesky decomposition.
    """
    rng_gp, rng_rest = random.split(rng)
    # NOTE: can't jit this; hlo lowering fails on cholesky?
    ls = Prior("beta", {"a": 3.0, "b": 7.0})
    var = Prior("fixed", {"value": 1.0})
    f_mu_s, _, ls, *_ = GP(kernel, var, ls, jitter=1e-5).simulate(rng_gp, s, batch_size)
    f, beta, f_obs_noise = _jax_helper(rng_rest, x, f_mu_s.squeeze())
    return f[..., None], ls, beta, f_obs_noise  # f: [B, L, 1]


@jit
def _jax_helper(rng: jax.Array, x: jax.Array, f_mu_s: jax.Array):
    B, (L, D) = f_mu_s.shape[0], x.shape
    rng_beta, rng_sigma, rng_noise = random.split(rng, 3)
    beta = random.normal(rng_beta, (B, D))
    f_mu_x = beta @ x.T  # [B, L]
    f_obs_noise = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))  # HalfNormal(0.1)
    f = f_mu_x + f_mu_s + f_obs_noise[:, None] * random.normal(rng_noise, (B, L))
    return f, beta, f_obs_noise


def build_dataloader(prior_pred: Callable, data: DictConfig):
    """Generates batches of model samples.

    Args:
        prior_pred: Callable of (x: [L, D], s: [L, S], batch_size) -> f: [B, L, 1]
    """
    B, S, D = data.batch_size, len(data.s), data.num_features
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num_test or all points
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = lambda x: jnp.repeat(x[None, ...], B, axis=0)

    def gen_batch(rng: jax.Array):
        rng_s, rng_x, rng_v = random.split(rng, 3)
        x = random.normal(rng_x, (L, D))  # features for every location
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        f, *rest = prior_pred(rng, x, s, B)  # x: [L, D], s: [L, S], f: [B, L, 1]
        x, s = batchify(x), batchify(s)  # x: [B, L, D], s: [B, L, S]
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = jnp.concatenate([s, x], axis=-1)  # [B, L, S + D]
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, *rest

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader
