from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.spatial import KDTree as spKDTree
from sps.kernels import l2_dist


@partial(jit, static_argnames=("k", "dist"))
def k_nearest_senders(
    r: jax.Array,
    s: jax.Array,
    k: int,
    dist: Callable = l2_dist,
):
    r"""Faster than scipy's `KDTree` for small tasks, but uses a $O(n^2)$ memory."""
    d = dist(r, s)
    idx = jnp.argsort(d, axis=-1)
    d = jnp.take_along_axis(d, idx, axis=-1)
    return idx[:, :k].flatten(), d[:, :k].flatten()


@partial(jit, static_argnames=("k",))
def scipy_k_nearest_senders(r: jax.Array, s: jax.Array, k: int):
    r"""Slower than JAX's O(n^2) implementation for small tasks, but scales in $O(N\log N)$."""
    d_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.float32)
    idx_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.int32)
    f = lambda r, s, k: spKDTree(np.array(s)).query(np.array(r), int(k))
    d, idx = jax.pure_callback(f, (d_shape, idx_shape), r, s, k)
    return idx.flatten(), d.flatten()
