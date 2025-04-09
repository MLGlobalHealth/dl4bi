from functools import partial
from typing import Callable

import jax
from jax import jit, vmap
import jax.numpy as jnp
from numpy import sort
from omegaconf import DictConfig, OmegaConf
import hashlib


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
    x_long, x_lat = x
    y_long, y_lat = y
    x_long, x_lat, y_long, y_lat = map(jnp.deg2rad, (x_long, x_lat, y_long, y_lat))

    return 2 * jnp.arcsin(
        jnp.sqrt(
            jnp.sin((x_lat - y_lat) / 2) ** 2
            + jnp.cos(x_lat) * jnp.cos(y_lat) * jnp.sin((x_long - y_long) / 2) ** 2
        )
    )


def make_pairwise(fn: Callable):
    return vmap(vmap(fn, in_axes=(None, 0)), in_axes=(0, None))


def cfg_to_run_name(cfg: DictConfig):
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.md5(cfg_str.encode()).hexdigest()


def cartesian_product(*xs):
    n = len(xs)
    return jnp.stack(jnp.meshgrid(*xs), axis=-1).reshape(-1, n)
