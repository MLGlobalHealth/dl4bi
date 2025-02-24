import jax
import jax.numpy as jnp
import tqdm
from jax import random

from dl4bi.meta_learning.autoregressive import (
    closest_first,
    furthest_first,
    invert_permutation,
    random_permutations,
)


def is_permutation(
    p: jax.Array,  # [N]
):
    return (jnp.sort(p) == jnp.arange(len(p), dtype=p.dtype)).all()


def random_tests(
    # Random tests
    N_tests=1000,
    L_ctx=3,
    L_test=10,
    B=2,  # batch size
    D=1,  # dimension of s (locations)
    rng=random.key(0),
):
    for i in tqdm.trange(N_tests):
        rng, rng_ctx, rng_test, rng_perm = random.split(rng, 4)
        s_ctx = random.beta(rng_ctx, 3, 7, (L_ctx, D))
        s_test = random.uniform(rng_test, (L_test, D))

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
        paths = random.normal(rng_test, (B, L_test, 1))

        idx = random_permutations(rng_perm, L_test, B)
        idx_inv = jax.vmap(invert_permutation)(idx)
        idx_inv_slow = jnp.stack([invert_permutation(idx[i]) for i in range(B)])
        assert (idx_inv == idx_inv_slow).all(), f"{idx_inv}\n\n{idx_inv_slow}"

        s_test_permuted = jnp.take_along_axis(
            s_test, jnp.repeat(idx[..., None], D, axis=-1), axis=1
        )
        s_test_permuted_slow = jnp.stack([s_test[i][idx[i]] for i in range(B)])
        for i in range(B):
            assert is_permutation(idx[i]), f"{idx[i]}"
            assert (s_test[i][idx[i]][idx_inv[i]] == s_test[i]).all(), (
                "ordering not preserved"
            )
        assert (s_test_permuted == s_test_permuted_slow).all()

        paths_inverted = jnp.take_along_axis(paths, idx_inv[..., None], axis=1)
        paths_inverted_permuted = jnp.take_along_axis(
            paths_inverted, idx[..., None], axis=1
        )
        assert (paths == paths_inverted_permuted).all()
        # end of snippet


if __name__ == "__main__":
    from time import time_ns

    s_ctx = jnp.array([0, 4, 8])[..., None]
    s_test = jnp.array([1, 2, 3, 5, 6, 7])[..., None]

    # manual test 1
    idx = furthest_first(s_ctx, s_test)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx] == jnp.array([2, 6, 1, 3, 5, 7])[..., None]).all()

    # manual test 2
    idx = closest_first(s_ctx, s_test)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx] == jnp.array([1, 2, 3, 5, 6, 7])[..., None]).all(), s_test[idx]

    # manual test 3
    s_ctx = jnp.array([1, 2, 3])[..., None]
    s_test = jnp.array([0.1, 4, 5, 6, 7, 7.1])[..., None]
    idx = closest_first(s_ctx, s_test)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx] == jnp.array([0.1, 4, 5, 6, 7, 7.1])[..., None]).all(), s_test[
        idx
    ]

    # manual test 4
    s_ctx = jnp.array([1, 2, 3])[..., None]
    s_test = jnp.array([1.1, 1.8, 1.5, 2.5])[..., None]
    idx = closest_first(s_ctx, s_test)
    assert is_permutation(idx), f"{idx}"
    assert (s_test[idx] == jnp.array([1.1, 1.8, 1.5, 2.5])[..., None]).all(), s_test[
        idx
    ]

    random_tests(rng=random.key(time_ns()), D=2)

    print("All tests passed.")
