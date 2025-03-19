from collections import defaultdict
from time import time

import jax.numpy as jnp
import numpy as np
from jax import random, vmap

from dl4bi.core.knn import (
    approx_knn,
    bf_knn,
    st_approx_knn,
    st_bf_knn,
)
from dl4bi.core.sim import l2_dist
from dl4bi.core.utils import mask_from_valid_lens
from dl4bi.meta_learning.sgnp import approx_knn


def test_approx_knn():
    B, L, D, K = 4, 10, 2, 3
    rng = random.key(42)
    q_s = r_s = random.normal(rng, (B, L, D))
    v_approx_knn = vmap(
        lambda **kwargs: approx_knn(**kwargs, k=K, s_sim=l2_dist, num_q_parallel=2)
    )
    idx, d_x, d_s, d_t = v_approx_knn(q_s=q_s, r_s=r_s)
    assert d_s.shape == (B, L, K)
    assert d_x is None
    assert d_t is None


def test_knn_shapes():
    rng = random.key(55)
    B, L, S, K = 1, 128, 2, 16
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, 1e6)
    idx, d = vmap(lambda q, r: bf_knn(q, r, K))(q, r)
    idx_a, d_a = vmap(lambda q, r: approx_knn(q, r, K))(q, r)
    idx_match_rate = (jnp.isin(idx, idx_a)).mean()
    assert idx.shape == (B, L, K), "Indices (true) are not the correct shape"
    assert idx_a.shape == (B, L, K), "Indices (approx) are not the correct shape"
    assert d.shape == (B, L, K, 1), "Distances (true) are not the correct shape"
    assert d_a.shape == (B, L, K, 1), "Distances (approx) are not the correct shape"
    assert idx_match_rate > 0.95, "Indices do not approximately match true<=>approx!"


def test_knn_speed():
    rng = random.key(55)
    B, L, S, K, N = 4, 1024, 2, 512, 10
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, 1e6)
    idx, d = vmap(lambda q, r: bf_knn(q, r, K))(q, r)
    idx_a, d_a = vmap(lambda q, r: approx_knn(q, r, K))(q, r)
    results = defaultdict(list)
    for name, method in [("bf", bf_knn), ("approx", approx_knn)]:
        t_start = time()
        for i in range(N):
            vmap(lambda q, r: method(q, r, K))(q, r)
        t_stop = time()
        results[name] += [t_stop - t_start]
    results = {k: np.mean(v) for k, v in results.items()}
    assert results["approx"] < results["bf"], "Approx kNN not faster than brute force!"


def test_st_knn_shapes():
    rng = random.key(55)
    B, L, S, K = 4, 128, 3, 16
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, jnp.inf)
    idx, d = vmap(lambda q, r: st_bf_knn(q, r, K))(q, r)
    idx_a, d_a = vmap(lambda q, r: st_approx_knn(q, r, K))(q, r)
    idx_match_rate = (jnp.isin(idx, idx_a)).mean()
    assert idx.shape == (B, L, K), "Indices (true) are not the correct shape"
    assert idx_a.shape == (B, L, K), "Indices (approx) are not the correct shape"
    assert d.shape == (B, L, K, 2), "Distances (true) are not the correct shape"
    assert d_a.shape == (B, L, K, 2), "Distances (approx) are not the correct shape"
    assert idx_match_rate > 0.95, "Indices do not approximately match true<=>approx!"


def test_st_knn_speed():
    rng = random.key(55)
    B, L, S, K, N = 4, 1024, 2, 512, 10
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, 1e6)
    idx, d = vmap(lambda q, r: st_bf_knn(q, r, K))(q, r)
    idx_a, d_a = vmap(lambda q, r: st_approx_knn(q, r, K))(q, r)
    results = defaultdict(list)
    for name, method in [("bf", bf_knn), ("approx", approx_knn)]:
        t_start = time()
        for i in range(N):
            vmap(lambda q, r: method(q, r, K))(q, r)
        t_stop = time()
        results[name] += [t_stop - t_start]
    results = {k: np.mean(v) for k, v in results.items()}
    assert results["approx"] < results["bf"], "Approx kNN not faster than brute force!"
