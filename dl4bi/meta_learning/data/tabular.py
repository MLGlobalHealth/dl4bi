from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
from jax import jit, random

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    permute_L_in_BLD,
)


@dataclass(frozen=True, eq=False)
class TabularData(MetaLearningData):
    """A simple `TabularData` container."""

    x: jax.Array  # [B, L, D_x]
    f: jax.Array  # [B, L, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
    ):
        return _batch(
            rng,
            self.x,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
        )


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
    ),
)
def _batch(
    rng: jax.Array,
    x: jax.Array,
    f: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool,
):
    rng_p, rng_b = random.split(rng)
    x, f, inv_permute_idx = permute_L_in_BLD(rng, [x, f])
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    args = batch_BLD(rng_b, [x, f], *batch_args)
    return TabularBatch(*args, inv_permute_idx=inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularData,
    lambda d: ((d.x, d.f), None),
    lambda _aux, children: TabularData(*children),
)


@dataclass(frozen=True, eq=False)
class TabularBatch(MetaLearningBatch):
    x_ctx: jax.Array
    f_ctx: jax.Array
    mask_ctx: Optional[jax.Array]
    x_test: jax.Array
    f_test: jax.Array
    mask_test: Optional[jax.Array] = None
    inv_permute_idx: Optional[jax.Array] = None


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularBatch,
    lambda d: (
        (
            d.x_ctx,
            d.f_ctx,
            d.mask_ctx,
            d.x_test,
            d.f_test,
            d.mask_test,
            d.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: TabularBatch(*children),
)
