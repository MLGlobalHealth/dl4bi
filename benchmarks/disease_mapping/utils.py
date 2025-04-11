import hashlib
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from numpyro.handlers import seed, substitute
from omegaconf import DictConfig, OmegaConf

from benchmarks.disease_mapping.model import jitter, kernel, prevalence
from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import TrainState


@partial(jit, static_argnames=["batch_size"])
def batch(x: jax.Array, batch_size: int):
    N, *dims = x.shape
    N_rounded = N // batch_size * batch_size
    if N != N_rounded:
        # this is ok being printed at compile time only
        print(
            f"batch_size does not divide the number of samples. Dropping final {N - N_rounded} samples."
        )
    x = x[:N_rounded].reshape(-1, batch_size, *dims)
    return x


@jit
def unbatch(x: jax.Array):
    N, B, *dims = x.shape
    x = x.reshape(-1, *dims)
    return x


def haversine_distance(x, y):
    """Haversine distance.

    x, y: two (Long, Lat) pairs in degrees
    """

    # based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    x_lon, x_lat = x
    y_lon, y_lat = y
    x_lon, x_lat, y_lon, y_lat = map(jnp.deg2rad, (x_lon, x_lat, y_lon, y_lat))

    arc_length = 2 * jnp.arcsin(
        jnp.sqrt(
            jnp.sin((x_lat - y_lat) / 2) ** 2
            + jnp.cos(x_lat) * jnp.cos(y_lat) * jnp.sin((x_lon - y_lon) / 2) ** 2
        )
    )

    return jnp.rad2deg(arc_length)


def great_circle_distance(x, y):
    # based on https://en.wikipedia.org/wiki/Great-circle_distance#Computational_formulae
    x_lon, x_lat = x
    y_lon, y_lat = y
    x_lon, x_lat, y_lon, y_lat = map(jnp.deg2rad, (x_lon, x_lat, y_lon, y_lat))

    d_lon = jnp.abs(x_lon - y_lon)

    sin = jnp.sin
    cos = jnp.cos

    arc_length = jnp.atan2(
        jnp.sqrt(
            (cos(y_lat) * sin(d_lon)) ** 2
            + (cos(x_lat) * sin(y_lat) - sin(x_lat) * cos(y_lat) * cos(d_lon)) ** 2
        ),
        sin(x_lat) * sin(y_lat) + cos(x_lat) * cos(y_lat) * cos(d_lon),
    )

    return jnp.rad2deg(arc_length)


def make_pairwise(fn: Callable):
    return vmap(vmap(fn, in_axes=(None, 0)), in_axes=(0, None))


def hash_config(cfg: DictConfig):
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()


def cartesian_product(*xs):
    n = len(xs)
    return jnp.stack(jnp.meshgrid(*xs), axis=-1).reshape(-1, n)


@jit
def sample_gp(
    rng,
    s_c: jax.Array,  # [B, L_ctx, D]
    y_c: jax.Array,  # [B, L_ctx]
    s_t: jax.Array,  # [B, L_test, D]
    **params,  # passes params to gp mean and kernel, each of dim [B, ...]
):
    """GP conditional sampling using Matheron's Rule."""
    B, L_ctx = y_c.shape
    _, L_test, _ = s_t.shape

    s = jnp.concat([s_c, s_t], axis=1)
    cov = vmap(kernel)(s, s, **params)
    cov += jitter * jnp.eye(L_ctx + L_test)

    L = jax.lax.linalg.cholesky(cov)
    z = jax.random.normal(rng, shape=(B, L_ctx + L_test))
    y = jnp.einsum("bij,bj->bi", L, z)

    cov_cc = cov[:, :L_ctx, :L_ctx]
    cov_tc = cov[:, L_ctx:, :L_ctx]

    return y[:, L_ctx:] + (
        cov_tc
        @ jsp.linalg.solve(cov_cc, (y_c - y[:, :L_ctx])[..., None], assume_a="pos")
    ).squeeze(-1)


def get_np_sampler(
    state: TrainState,
):
    @jit
    def sample_np(
        rng,
        s_c: jax.Array,  # [B, L, D]
        y_c: jax.Array,  # [B, L]
        s_t: jax.Array,  # [B, L, D]
        **params,  # ignored
    ):
        """Batched Neural Process sampler"""
        rng, rng_extra = jax.random.split(rng)
        output = state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_c,
            y_c[:, None],
            s_t,
            None,
            rngs={"extra": rng_extra},
        )

        if isinstance(output, tuple):
            output = output[0]

        match output:
            case DiagonalMVNOutput(mu, std):
                assert mu.ndim == 2 and std.ndim == 2

                return mu + std * jax.random.normal(rng, mu.shape)
            case _:
                raise NotImplementedError()

    return sample_np


@jit
def sample_prevalence(rng, y, **params):
    """Batched sampling of prevalence given `y` and `params`.
    Args:
        rng: jax.random.PRNGKey
        y: spatial effect
        params: params to be plugged into the prevalence model.

    Returns:
        logit(prevalence)
    """
    rng = jax.random.split(rng, y.shape[0])
    return vmap(lambda rng, y, params: seed(substitute(prevalence, params), rng)(y))(
        rng, y, params
    )
