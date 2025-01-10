from collections.abc import Callable
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.spatial import KDTree as spKDTree
from sps.kernels import l2_dist


@partial(jit, static_argnames=("k", "dist", "batch_size"))
def k_nearest_senders(
    r: jax.Array,
    s: jax.Array,
    k: int,
    dist: Callable = l2_dist,
    batch_size: Optional[int] = None,
):
    r"""Parellelized brute force kNN with optional `batch_size`.

    Args:
        r: Reciever positions.
        s: Sender positions.
        k: Number of neighbors per receiver.
        dist: Distance function to use.
        batch_size: Number of receivers to run in parallel. By default,
            the method runs all of them, i.e. is a `vmap`.

    Returns:
        Index and distance arrays, each of dimension |r| x k.
    """
    if batch_size is None:
        batch_size = r.shape[0]

    def process_batch(r_i: jax.Array):
        # add leading dim to r_i since map processes each r_i individually,
        # even when batch_size is >= 1
        d = dist(r_i[None, ...], s)
        idx = jnp.argsort(d, axis=-1)
        d = jnp.take_along_axis(d, idx, axis=-1)
        return idx[:, :k].flatten(), d[:, :k].flatten()

    return jax.lax.map(process_batch, r, batch_size=batch_size)


@partial(jit, static_argnames=("k",))
def scipy_k_nearest_senders(r: jax.Array, s: jax.Array, k: int):
    r"""Slower than JAX's O(n^2) implementation for small tasks, but scales in $O(N\log N)$."""
    d_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.float32)
    idx_shape = jax.ShapeDtypeStruct((r.shape[0], k), jnp.int32)
    f = lambda r, s, k: spKDTree(np.array(s)).query(np.array(r), int(k))
    d, idx = jax.pure_callback(f, (d_shape, idx_shape), r, s, k)
    return idx.flatten(), d.flatten()
