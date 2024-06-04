from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training import train_state
from jax import jit, lax
from jax._src.numpy.util import promote_dtypes_inexact


class TrainState(train_state.TrainState):
    kwargs: FrozenDict = FrozenDict({})


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    return (jnp.arange(max_len) < valid_lens[..., None])[..., None]


@jit
def l2_dist_sq(x: jax.Array, y: jax.Array) -> jax.Array:
    """L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = prepare_dims(x, y)
    return (x**2).sum(-1)[:, None] + (y**2).sum(-1).T - 2 * x @ y.T


@jit
def prepare_dims(x: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Two `[N, D]` dimensional arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y


def mvn_logpdf_tril_cov(x: jax.Array, mean: jax.Array, L: jax.Array):
    """MVN logpdf with lower triangular covariance.

    Based on jax implementation [here](https://github.com/google/jax/blob/main/ jax/_src/scipy/stats/multivariate_normal.py#L25-L73).
    """
    x, mean, L = promote_dtypes_inexact(x, mean, L)
    n = mean.shape[-1]
    y = jnp.vectorize(
        partial(lax.linalg.triangular_solve, lower=True, transpose_a=True),
        signature="(n,n),(n)->(n)",
    )(L, x - mean)
    return (
        -1 / 2 * jnp.einsum("...i,...i->...", y, y)
        - n / 2 * jnp.log(2 * np.pi)
        - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
    )
