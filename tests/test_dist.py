import jax.numpy as jnp
from jax import random, vmap

from dl4bi.core import knn, mask_from_valid_lens, scipy_knn


def test_knn():
    rng = random.key(55)
    B, L, S, K = 4, 128, 2, 16
    q = random.normal(rng, (B, L, S))
    v = random.randint(rng, (B,), 0, L)
    m = mask_from_valid_lens(L, v)
    r = jnp.where(m, q, 1e6)
    idx, d = vmap(lambda q, r: knn(q, r, K))(q, r)
    idx_s, d_s = vmap(lambda q, r: scipy_knn(q, r, K))(q, r)
    assert (idx == idx_s).all(), "Indices do not match!"
    assert jnp.allclose(d, d_s), "Distances are not close!"
