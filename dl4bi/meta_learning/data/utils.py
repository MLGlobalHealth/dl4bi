"""Shared batching helpers for meta-learning datasets."""

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import jit, random, vmap

from ...core.data import Batch, Data, ElementSelectorMixin
from ...core.utils import mask_from_valid_lens, nan_pad


class MetaLearningData(Data, ElementSelectorMixin):
    """Base container for meta-learning datasets."""

    pass


class MetaLearningBatch(Batch, ElementSelectorMixin):
    """Base container for sampled meta-learning batches."""

    pass


@partial(jit, static_argnames=("independent",))
def permute_L_in_BLD(
    rng: jax.Array,
    arrays: Sequence[jax.Array],
    independent: bool = False,
):
    """Permute the sequence axis of ``[B, L, D]`` arrays.

    Args:
        rng: PRNG key used to draw permutations.
        arrays: Arrays that share the same leading ``[B, L]`` dimensions.
        independent: Whether to sample a different permutation per batch item.

    Returns:
        The permuted arrays followed by the inverse permutation indices.
    """
    B, L = arrays[0].shape[:2]
    if independent:
        rngs = random.split(rng, B)
        permute_idx = _vpermute_idx(rngs, L)
        inv_permute_idx = jnp.argsort(permute_idx, axis=1)
        vpermute = vmap(lambda v, idx: v[idx])
        arrays = [vpermute(a, permute_idx) for a in arrays]
        return *arrays, inv_permute_idx
    permute_idx = _permute_idx(rng, L)
    inv_permute_idx = jnp.argsort(permute_idx)
    arrays = [a[:, permute_idx] for a in arrays]
    return *arrays, inv_permute_idx


def _vpermute_idx(rngs: jax.Array, L: int):
    """Vectorize permutation sampling over a batch of PRNG keys."""
    return vmap(lambda rng: _permute_idx(rng, L))(rngs)


def _permute_idx(rng: jax.Array, L: int):
    """Sample a permutation of ``range(L)``."""
    return random.choice(rng, L, (L,), replace=False)


@jit
def inv_permute_L_in_BLD(
    arrays: Sequence[jax.Array],
    inv_permute_idx: jax.Array,
):
    """Undo a permutation previously returned by :func:`permute_L_in_BLD`."""
    if inv_permute_idx.ndim == 1:
        return [a[:, inv_permute_idx] for a in arrays]
    v_inv = vmap(lambda v, idx: v[idx])
    return [v_inv(a, inv_permute_idx) for a in arrays]


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
    ),
)
def batch_BLD(
    rng: jax.Array,
    arrays: Sequence[jax.Array],
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool = False,
):
    """Split ``[B, L, D]`` arrays into context and test sets."""
    B = arrays[0].shape[0]
    valid_lens_ctx = random.randint(rng, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    ctx = [a[:, :num_ctx_max] for a in arrays]
    start = 0 if test_includes_ctx else num_ctx_max
    test = [a[:, start : start + num_test] for a in arrays]
    return (
        *ctx,
        mask_from_valid_lens(num_ctx_max, valid_lens_ctx),
        *test,
        mask_from_valid_lens(num_test, valid_lens_test),
    )


@partial(jit, static_argnames=("L",))
def unbatch_BLD(arrays: Sequence[jax.Array], L: int):
    """Pad variable-length ``[B, L_i, D]`` arrays back to a common length."""
    return [nan_pad(a, axis=1, L=L) for a in arrays]
