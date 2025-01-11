from collections.abc import Callable
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from scipy.spatial import KDTree as spKDTree
from sps.kernels import l2_dist


class kNN(nn.Module):
    r"""Parellelized brute force kNN with optional `batch_size`.

    Args:
        k: Number of neighbors per query point to retrieve.
        dist: Distance function to use.
        num_q_parallel: Number of queries to run in parallel for each element in
            the batch.

    Returns:
        An instance of `kNN`.
    """

    k: int = 16
    dist: Callable = l2_dist
    num_q_parallel: int = 1024

    @nn.compact
    def __call__(self, q, r):
        vknn = vmap(lambda q, r: bf_knn(q, r, self.k, self.dist, self.num_q_parallel))
        return vknn(q, r)


@partial(jit, static_argnames=("k", "dist", "num_q_parallel"))
def bf_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    dist: Callable = l2_dist,
    num_q_parallel: int = 1024,
):
    r"""Parellelized brute force kNN with optional `batch_size`.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
        dist: Distance function to use.
        batch_size: Number of queries to run in parallel.

    Returns:
        Index and distance arrays, each of dimension |r| x k.
    """

    def process_batch(q_i: jax.Array):
        # add leading dim to q_i since map processes each q_i individually,
        # even when batch_size is >= 1
        d = dist(q_i[None, ...], r)
        idx = jnp.argsort(d, axis=-1)
        d = jnp.take_along_axis(d, idx, axis=-1)
        return idx[:, :k].flatten(), d[:, :k]

    idx, d = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, d.squeeze()  # d: [B, L, 1, K] -> [B, L, K]


@partial(jit, static_argnames=("k",))
def scipy_knn(q: jax.Array, r: jax.Array, k: int):
    r"""Slower than JAX's O(n^2) implementation for small tasks, but scales in $O(N\log N)$."""
    d_shape = jax.ShapeDtypeStruct((q.shape[0], k), jnp.float32)
    idx_shape = jax.ShapeDtypeStruct((q.shape[0], k), jnp.int32)
    f = lambda q, r, k: spKDTree(np.array(r)).query(np.array(q), int(k))
    d, idx = jax.pure_callback(f, (d_shape, idx_shape), q, r, k)
    return idx, d
