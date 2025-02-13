import jax
import jax.numpy as jnp
from jax import random

from autoregressive import furthest_first, closest_first, invert_permutation


def is_permutation(p: jax.Array):
    return (jnp.sort(p) == jnp.arange(len(p), dtype=p.dtype)).all()


s_ctx = jnp.array([0, 4, 8])[:, None]
s_test = jnp.array([1, 2, 3, 5, 6, 7])[:, None]

# manual test 1
idx = furthest_first(s_ctx, s_test)
assert is_permutation(idx), f"{idx}"
assert (s_test[idx] == jnp.array([2, 6, 1, 3, 5, 7])[:, None]).all()

# manual test 2
idx = closest_first(s_ctx, s_test)
assert is_permutation(idx), f"{idx}"
assert (s_test[idx] == jnp.array([1, 2, 3, 5, 6, 7])[:, None]).all(), s_test[idx]


# Random tests
N_tests = 1000
L_ctx = 3
L_test = 10
B = 2  # batch size
D_s = 1  # dimension of s (locations)
D_f = 3  # dimension of f (values)
rng = random.key(42)

for i in range(N_tests):
    rng, rng_ctx, rng_test, rng_perm = random.split(rng, 4)
    s_ctx = random.beta(rng_ctx, 3, 7, (L_ctx, D_s))
    s_test = random.uniform(rng_test, (L_test, D_s))

    # test furthest
    idx = furthest_first(s_ctx, s_test)
    idx_inv = invert_permutation(idx)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx][idx_inv] == s_test).all(), "ordering not preserved"

    # test closest
    idx = closest_first(s_ctx, s_test)
    idx_inv = invert_permutation(idx)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx][idx_inv] == s_test).all(), "ordering not preserved"

    # test random permutation
    s_test = jnp.repeat(s_test[None], B, axis=0)
    paths = random.normal(rng_test, (B, L_test, D_f))

    # snippet from autoregressive.py
    idx = jnp.repeat(jnp.arange(L_test)[None, :], B, axis=0)
    idx = random.permutation(rng_perm, idx, axis=1, independent=True)
    idx_inv = jax.vmap(invert_permutation)(idx)

    for i in range(B):
        assert is_permutation(idx[i]), f"{idx[i]}"
        assert (
            s_test[i][idx[i]][idx_inv[i]] == s_test[i]
        ).all(), "ordering not preserved"

    s_test_permuted = jnp.zeros_like(s_test)
    for i in range(B):
        s_test_permuted = s_test_permuted.at[i].set(s_test[i][idx[i]])

    idx = jnp.repeat(
        idx[:, :, None], D_s, axis=2
    )  # needs to match dimension of s_test in take_along_axis
    s_test_permuted_with_take_along_axis = jnp.take_along_axis(s_test, idx, axis=1)
    assert (
        s_test_permuted == s_test_permuted_with_take_along_axis
    ).all(), f"take_along_axis doesn't work as expected\n{s_test_permuted.squeeze()}\n\n{s_test_permuted_with_take_along_axis.squeeze()}"
    # end of snippet


print("All tests passed.")
