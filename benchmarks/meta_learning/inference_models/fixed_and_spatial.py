from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit, random, vmap
from numpyro.infer import Predictive
from omegaconf import DictConfig
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from sps.utils import build_grid

from .utils import condition_gp


def numpyro_model(
    x_ctx: jax.Array,  # [L, D]
    s_ctx: jax.Array,  # [L, S]
    f_ctx: jax.Array,  # [L, 1]
    jitter: float = 1e-5,
    **kwargs,
):
    """Generic Spatiotemporal model with fixed and random spatial effects.

    Args:
        x: Array of input covariates, `[L, D]`.
        s: Array of input locations, `[L, S]`.
        f: Observed function values, `[L, 1]`.
    """

    L, D = x_ctx.shape
    var = numpyro.deterministic("var", 1.0)
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L)
    beta = numpyro.sample("beta", dist.Normal(jnp.zeros(D), jnp.ones(D)))
    f_mu_x = x_ctx @ beta
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(jnp.zeros(L), k))
    f_mu = f_mu_x + f_mu_s
    f_obs_noise = numpyro.sample("f_obs_noise", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu, f_obs_noise), obs=f_ctx)


@partial(jit, static_argnames=("batch_size",))
def numpyro_prior_pred(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    B = batch_size
    prior_pred = Predictive(numpyro_model, num_samples=B)
    return prior_pred(rng, x=x, s=s)


@partial(jit, static_argnames=("jitter",))
def numpyro_pointwise_post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, S]
    s_test: jax.Array,  # [L_test, S]
    x_test: jax.Array,  # [L_test, D]
    beta: jax.Array,  # [N, L_test, D]
    var: jax.Array,  # [N]
    ls: jax.Array,  # [N]
    f_mu_s: jax.Array,  # [N, L_ctx]
    f_obs_noise: jax.Array,  # [N]
    jitter: float = 1e-5,
    **kwargs,
):
    """Calculates the pointwise posterior predictive."""
    N = ls.shape[0]
    f = vmap(numpyro_post_pred, in_axes=(0, None, None, None, 0, 0, 0, 0, 0, None))(
        random.split(rng, N),
        s_ctx,
        s_test,
        x_test,
        beta,
        var,
        ls,
        f_mu_s,
        f_obs_noise,
        jitter,
    )  # f: [N, L]
    return f.mean(axis=0), f.std(axis=0)


@partial(jit, static_argnames=("jitter",))
def numpyro_post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, S]
    s_test: jax.Array,  # [L_test, S]
    x_test: jax.Array,  # [L_ctx, D]
    beta: jax.Array,  # [D]
    var: jax.Array,  # [1]
    ls: jax.Array,  # [1]
    f_mu_s: jax.Array,  # [L_ctx, 1]
    f_obs_noise: jax.Array,  # [1]
    jitter: float = 1e-5,
):
    """Generates a posterior predictive sample.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    rng_gp, rng_noise = random.split(rng)
    f_mu = condition_gp(rng_gp, s_ctx, f_mu_s, s_test, var, ls, rbf, jitter)
    f_mu += beta @ x_test.T
    f = f_mu + f_obs_noise * random.normal(rng_noise, f_mu.shape)
    return f


def jax_prior_pred(
    rng: jax.Array,
    x: jax.Array,  # [L, D]
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
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
    f_mu_s, var, ls, *_ = GP(rbf, var, ls, jitter=1e-5).simulate(rng_gp, s, batch_size)
    f_mu_s = f_mu_s.squeeze()
    f, beta, f_obs_noise = _jax_helper(rng_rest, x, f_mu_s)
    return {
        "var": var,
        "ls": ls,
        "beta": beta,
        "f_mu_s": f_mu_s,
        "f_obs_noise": f_obs_noise,
        "f": f,
    }


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
        samples = prior_pred(rng, x, s, B)  # x: [L, D], s: [L, S], f: [B, L, 1]
        f = samples["f"][..., None]  # [B, L, 1]
        x, s = batchify(x), batchify(s)  # x: [B, L, D], s: [B, L, S]
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = jnp.concatenate([s, x], axis=-1)  # [B, L, S + D]
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, samples

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def batch_to_infer_kwargs(batch: tuple, data: DictConfig, infer: DictConfig):
    S, Nc = len(data.s), infer.num_ctx
    _, _, _, s, f, *_ = batch
    x, s, f = s[0, :, S:], s[0, :, :S], f[0]
    return {
        "x_ctx": x[:Nc],
        "x_test": x[Nc:],
        "s_ctx": s[:Nc],
        "s_test": s[Nc:],
        "f_ctx": f[:Nc],
        "f_test": f[Nc:],
    }
