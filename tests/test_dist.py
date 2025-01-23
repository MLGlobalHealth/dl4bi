from collections import defaultdict
from time import time

import jax.numpy as jnp
import numpy as np
from jax import random, vmap

from dl4bi.core import approx_knn, bf_knn, mask_from_valid_lens, scipy_knn


def test_knn():
    rng = random.key(55)
    B, L, S, K = 4, 128, 2, 16
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, 1e6)
    idx, d = vmap(lambda q, r: bf_knn(q, r, K))(q, r)
    idx_s, d_s = vmap(lambda q, r: scipy_knn(q, r, K))(q, r)
    idx_a, d_a = vmap(lambda q, r: approx_knn(q, r, K))(q, r)
    assert (idx == idx_s).all(), "Indices do not match true<=>scipy!"
    assert (idx == idx_a).all(), "Indices do not match true<=>approx!"
    assert jnp.allclose(d, d_s), "Distances are not close true<=>scipy!"
    assert jnp.allclose(d, d_a), "Distances are not close true<=>approx!"


def test_approx_knn_speed():
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
