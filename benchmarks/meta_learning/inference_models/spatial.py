from functools import partial
from typing import Callable, Optional

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
    s_ctx: jax.Array,
    f_ctx: Optional[jax.Array] = None,
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

    L = s_ctx.shape[0]
    var = numpyro.deterministic("var", 1.0)
    ls = numpyro.sample("ls", dist.Beta(3, 7))
    k = rbf(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L)
    f_mu_s = numpyro.sample("f_mu_s", dist.MultivariateNormal(jnp.zeros(L), k))
    f_obs_noise = numpyro.sample("f_obs_noise", dist.HalfNormal(0.1))
    numpyro.sample("f", dist.Normal(f_mu_s, f_obs_noise), obs=f_ctx)


@partial(jit, static_argnames=("batch_size",))
def numpyro_prior_pred(
    rng: jax.Array,
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    B = batch_size
    prior_pred = Predictive(numpyro_model, num_samples=B)
    return prior_pred(rng, s=s)


@partial(jit, static_argnames=("jitter",))
def numpyro_pointwise_post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, S]
    s_test: jax.Array,  # [L_test, S]
    var: jax.Array,  # [N]
    ls: jax.Array,  # [N]
    f_mu_s: jax.Array,  # [N, L_ctx]
    f_obs_noise: jax.Array,  # [N]
    jitter: float = 1e-5,
    **kwargs,
):
    """Calculates the pointwise posterior predictive."""
    N = ls.shape[0]
    f = vmap(numpyro_post_pred, in_axes=(0, None, None, 0, 0, 0, 0, None))(
        random.split(rng, N), s_ctx, s_test, var, ls, f_mu_s, f_obs_noise, jitter
    )  # f: [N, L]
    return f.mean(axis=0), f.std(axis=0)


@partial(jit, static_argnames=("jitter",))
def numpyro_post_pred(
    rng: jax.Array,
    s_ctx: jax.Array,  # [L_ctx, S]
    s_test: jax.Array,  # [L_test, S]
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
    f = f_mu + f_obs_noise * random.normal(rng_noise, f_mu.shape)
    return f


def jax_prior_pred(
    rng: jax.Array,
    s: jax.Array,  # [L, S]
    batch_size: int = 64,
):
    """A pure JAX spatiotemporalmodel with random spatial effects.

    Technically this isn't the same model since the GP class samples
    the GP priors once per batch in order to amortize the cost of
    the Cholesky decomposition. However, it is usually at least 2x
    faster than sampling from the Numpyro model.
    """
    rng_gp, rng_rest = random.split(rng)
    # NOTE: can't jit this; hlo lowering fails on cholesky?
    ls = Prior("beta", {"a": 3.0, "b": 7.0})
    var = Prior("fixed", {"value": 1.0})
    f_mu_s, var, ls, *_ = GP(rbf, var, ls, jitter=1e-5).simulate(rng_gp, s, batch_size)
    f, f_obs_noise = _jax_helper(rng_rest, f_mu_s.squeeze())
    return {"f": f, "var": var, "ls": ls, "f_obs_noise": f_obs_noise}


@jit
def _jax_helper(rng: jax.Array, f_mu_s: jax.Array):
    B, L = f_mu_s.shape
    rng_sigma, rng_noise = random.split(rng)
    f_obs_noise = 0.1 * jnp.abs(random.normal(rng_sigma, (B,)))  # HalfNormal(0.1)
    f = f_mu_s + f_obs_noise[:, None] * random.normal(rng_noise, (B, L))
    return f, f_obs_noise


def build_dataloader(prior_pred: Callable, data: DictConfig):
    """Generates batches of model samples.

    Args:
        prior_pred: Callable of (x: [L, D], s: [L, S], batch_size) -> f: [B, L, 1]
    """
    B, S = data.batch_size, len(data.s)
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num_test or all points
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = lambda x: jnp.repeat(x[None, ...], B, axis=0)

    def gen_batch(rng: jax.Array):
        rng_s, rng_v = random.split(rng, 2)
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        samples = prior_pred(rng, s, B)  # x: [L, D], s: [L, S], f: [B, L, 1]
        f = samples["f"][..., None]  # [B, L, 1]
        s = batchify(s)  # [B, L, S]
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, samples

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def batch_to_infer_kwargs(batch: tuple, data: DictConfig, infer: DictConfig):
    Nc = infer.num_ctx
    _, _, _, s, f, *_ = batch
    s, f = s[0], f[0]
    return {
        "s_ctx": s[:Nc],
        "s_test": s[Nc:],
        "f_ctx": f[:Nc],
        "f_test": f[Nc:],
    }
