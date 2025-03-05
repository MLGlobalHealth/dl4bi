from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax import jit, random, vmap

from ...core.data import Batch, Data, ElementSelectorMixin
from ...core.utils import mask_from_valid_lens, nan_pad


class MetaLearningData(Data, ElementSelectorMixin):
    pass


class MetaLearningBatch(Batch, ElementSelectorMixin):
    pass


@partial(jit, static_argnames=("independent",))
def permute_L_in_BLD(
    rng: jax.Array,
    arrays: Sequence[jax.Array],
    independent: bool = False,
):
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
    return vmap(lambda rng: _permute_idx(rng, L))(rngs)


def _permute_idx(rng: jax.Array, L: int):
    return random.choice(rng, L, (L,), replace=False)


@jit
def inv_permute_L_in_BLD(
    arrays: Sequence[jax.Array],
    inv_permute_idx: jax.Array,
):
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
    B = arrays[0].shape[0]
    valid_lens_ctx = random.randint(rng, (B,), num_ctx_min, num_ctx_max)
    valid_lens_test = jnp.repeat(num_test, B)
    ctx = [a[:, :num_ctx_max] for a in arrays]
    Nc, Nt = num_ctx_max, num_test
    if test_includes_ctx:
        test = [a[:, :Nt] for a in arrays]
    else:
        test = [a[:, Nc : Nc + Nt] for a in arrays]
    return (
        *ctx,
        mask_from_valid_lens(num_ctx_max, valid_lens_ctx),
        *test,
        mask_from_valid_lens(num_test, valid_lens_test),
    )


@partial(jit, static_argnames=("L",))
def unbatch_BLD(arrays: Sequence[jax.Array], L: int):
    return [nan_pad(a, axis=1, L=L) for a in arrays]
