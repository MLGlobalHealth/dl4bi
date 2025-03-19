from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap

from .sim import delta_time, l2_dist


@partial(jit, static_argnames=("k", "num_q_parallel", "recall_target"))
def approx_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    num_q_parallel: int = 1024,
    recall_target: float = 0.95,
):
    r"""Parellelized approximate kNN.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
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
        d = l2_dist(q_i[None, :], r).flatten()
        d, idx = jax.lax.approx_min_k(d, k, recall_target=recall_target)
        return idx, d[:, None]

    idx, d = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, {"d": d}


@partial(jit, static_argnames=("k", "num_q_parallel"))
def bf_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
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
        d = l2_dist(q_i[None, ...], r).flatten()
        idx = jnp.argsort(d)[:k]
        return idx, d[idx, None]

    idx, d = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, {"d": d}


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
    method: Callable = approx_knn
    num_q_parallel: int = 1024

    def __call__(
        self,
        q: jax.Array,  # [B, Q, D]
        r: jax.Array,  # [B, R, D]
        **kwargs,
    ):
        f = lambda q, r: self.method(q, r, self.k, self.num_q_parallel)
        return jit(vmap(f))(q, r)


@partial(jit, static_argnums=list(range(2, 7)))
def st_approx_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    scale_t: float = 1.0,
    causal_t: bool = False,
    num_q_parallel: int = 1024,
    recall_target: float = 0.95,
):
    r"""Parellelized spatiotemporal approximate kNN.

    This calculation proceeds as follows:

    1. The input arrays `q` and `r` are of shape `[Q, R, D]` where the first D-1
        dimensions are the spatial locations and the last dimension D is time.

    2. A single distance metric is composed as `d_s^2 + (scale_t * d_t)^2` and
        passed to the kNN module to find the nearest k neighbors in spacetime.
        `scale_t` allows the user to specify a tradeoff between spatial and
        temporal proximity.

    3. The function returns the indices of the k nearest neighbors as
        determined by the metric in 2., but returns the original distances of
        shape `[Q, K, 2]` so that the original spatial and temporal distances
        can be used independently by downstream bias modules. `[Q, K, 0]` is
        the spatial distance and `[Q, K, 1]` is the temporal distance.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
        scale_t: Scale time dimension before selecting neighbors.
        causal_t: Enforce temporal casuality.
        num_q_parallel: Number of queries to run in parallel.
        recall_target: Target percent of returned k values
            are actually in top-k. Less than 1.0 can result in
            much faster runtimes.

    Returns:
        `idx` and `d` arrays of shapes `[Q, K]` and `[Q, K, D]`.
    """

    def process_batch(q_i: jax.Array):
        # add leading dim to q_i since map processes each q_i individually,
        # even when batch_size is >= 1
        q_i = q_i[None, ...]
        d_s = l2_dist(q_i[..., :-1], r[..., :-1]).squeeze()  # [R]
        d_t = delta_time(q_i[..., [-1]], r[..., [-1]], causal_t).squeeze()  # [R]
        d_sq = d_s**2 + (scale_t * d_t) ** 2
        _, idx = jax.lax.approx_min_k(d_sq, k, recall_target=recall_target)
        return idx, d_s[idx], d_t[idx]

    idx, d_s, d_t = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, {"d_s": d_s, "d_t": d_t}


@partial(jit, static_argnums=list(range(2, 6)))
def st_bf_knn(
    q: jax.Array,
    r: jax.Array,
    k: int,
    scale_t: float = 1.0,
    causal_t: bool = True,
    num_q_parallel: int = 1024,
):
    r"""Parellelized spatiotemporal brute force kNN.

    This calculation proceeds as follows:

    1. Calculates the spatiotemporal distance, which returns an array of shape
        `[Q, R, 2]` where `[Q, R, 0]` represent the spatial L2 distances and
        `[Q, R, 1]` represent signed L1 temporal distances.

    2. A single distance metric is composed as `d_s^2 + (scale_t * d_t)^2` and
        passed to the kNN module to find the nearest k neighbors in spacetime.
        `scale_t` allows the user to specify a tradeoff between spatial and
        temporal proximity.

    3. The function returns the indices of the k nearest neighbors as
        determined by the metric in 2., but returns the original distances of
        shape `[Q, K, 2]` so that the original spatial and temporal distances
        can be used independently by downstream bias modules.

    Args:
        q: Query points.
        r: Reference points.
        k: Number of neighbors per query point to retrieve.
        scale_t: Scale time dimension before selecting neighbors.
        causal_t: Enforce temporal casuality.
        num_q_parallel: Number of queries to run in parallel.

    Returns:
        `idx` and `d` arrays of shapes `[Q, K]` and `[Q, K, D]`.
    """

    def process_batch(q_i: jax.Array):
        # add leading dim to q_i since map processes each q_i individually,
        # even when batch_size is >= 1
        q_i = q_i[None, ...]
        d_s = l2_dist(q_i[..., :-1], r[..., :-1]).squeeze()  # [R]
        d_t = delta_time(q_i[..., [-1]], r[..., [-1]], causal_t).squeeze()  # [R]
        d_sq = d_s**2 + (scale_t * d_t) ** 2
        idx = jnp.argsort(d_sq)[:k]
        return idx, d_s[idx], d_t[idx]

    idx, d_s, d_t = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, {"d_s": d_s, "d_t": d_t}


@dataclass(frozen=True)
class STkNN:
    r"""Parellelized Spatiotemporal kNN.

    Args:
        k: Number of neighbors per query point to retrieve.
        scale_t: Scale the time dimension when retrieving kNN neighbors in spacetime.
        causal_t: Enforce temporal causality.
        num_q_parallel: Number of queries to run in parallel for each element in
            the batch.

    Returns:
        An instance of `kNN`.
    """

    k: int = 16
    method: Callable = st_approx_knn
    scale_t: float = 1.0
    causal_t: bool = True
    num_q_parallel: int = 1024

    def __call__(
        self,
        q: jax.Array,  # [B, Q, D]
        r: jax.Array,  # [B, Q, D]
        **kwargs,
    ):
        f = lambda q, r: self.method(
            q,
            r,
            self.k,
            self.scale_t,
            self.causal_t,
            self.num_q_parallel,
        )
        return jit(vmap(f))(q, r)  # idx: [B, Q, K], d: [B, Q, K]
