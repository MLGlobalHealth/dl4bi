from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import l2_dist


@partial(jit, static_argnames=("k", "dist", "num_q_parallel"))
def approx_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    dist: Callable = l2_dist,
    num_q_parallel: int = 1024,
    recall_target: float = 0.95,
):
    r"""Parellelized approximate kNN.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
        dist: Distance function to use.
        num_q_parallel: Number of queries to run in parallel.
        recall_target: Target percent of returned k values
            are actually in top-k. Less than 1.0 can result in
            much faster runtimes.


    Returns:
        Index and distance arrays, each of dimension |r| x k.
    """

    def process_batch(q_i: jax.Array):
        # add leading dim to q_i since map processes each q_i individually,
        # even when batch_size is >= 1
        d = dist(q_i[None, ...], r)
        d, idx = jax.lax.approx_min_k(d, k, recall_target=recall_target)
        return idx.flatten(), d

    idx, d = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, jnp.swapaxes(d, -2, -1)  # d: [B, Q, 1, R] -> [B, Q, R, 1]


@partial(jit, static_argnames=("k", "dist", "num_q_parallel"))
def bf_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    dist: Callable = l2_dist,
    num_q_parallel: int = 1024,
):
    r"""Parellelized brute force kNN.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
        dist: Distance function to use.
        num_q_parallel: Number of queries to run in parallel.

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
    return idx, jnp.swapaxes(d, -2, -1)  # d: [B, Q, 1, R] -> [B, Q, R, 1]


@dataclass(frozen=True)
class kNN:
    r"""Parellelized kNN.

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
    method: Callable = approx_knn

    def __call__(self, q, r):
        f = lambda q, r: self.method(q, r, self.k, self.dist, self.num_q_parallel)
        return jit(vmap(f))(q, r)
