from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax import jit, random

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    permute_L_in_BLD,
)


@dataclass(frozen=True, eq=False)
class TabularData(MetaLearningData):
    """A simple `TabularData` container.

    This container separates features into a dict of `feature_groups` in case
    the grouping determines downstream behavior, e.g. different preprocessing or
    bias terms used by models.

    .. note::
        This class handles optional spatial and temporal data differently from
        `SpatiotemporalData` in that it doesn't group data by the timestamp,
        which is particularly useful when you have observations over time but
        they aren't synchronized across spatial locations.
    """

    # {'x': [B, L, D_x], 's': [B, L, D_s], 't': [B, L, D_t], ...}
    feature_groups: FrozenDict[str, jax.Array]
    f: jax.Array  # [B, L, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
        forecast: bool = False,  # requires a 't' feature group
        t_sorted: bool = False,  # requires a 't' feature group
    ):
        return _batch(
            rng,
            self.feature_groups,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
            forecast,
            t_sorted,
        )


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
        "forecast",
        "t_sorted",
    ),
)
def _batch(
    rng: jax.Array,
    feature_groups: FrozenDict[str, jax.Array],
    f: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool,
    forecast: bool,
    t_sorted: bool,
):
    rng_p, rng_b = random.split(rng)
    prepare_L_in_BLD = permute_L_in_BLD
    if forecast:
        L = f.shape[1]
        inv_permute_idx = jnp.arange(L)
        prepare_L_in_BLD = jit(lambda _rng, arrays: (*arrays, inv_permute_idx))
        if not t_sorted:
            sort_idx = jnp.argsort(feature_groups["t"], axis=1)
            d = {
                k: jnp.take_along_axis(v, sort_idx, axis=1)
                for k, v in feature_groups.items()
            }
            feature_groups = FrozenDict(d)
            f = jnp.take_along_axis(f, sort_idx, axis=1)
    *arrays, inv_permute_idx = prepare_L_in_BLD(rng, [*feature_groups.values(), f])
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    args = batch_BLD(rng_b, arrays, *batch_args)
    num_ctx_args = len(args) // 2
    ctx, test = args[:num_ctx_args], args[num_ctx_args:]
    *ctx, mask_ctx = ctx
    *test, mask_test = test
    names = list(feature_groups) + ["f"]
    ctx = FrozenDict(zip([f"{name}_ctx" for name in names], ctx))
    test = FrozenDict(zip([f"{name}_test" for name in names], test))
    return TabularBatch(ctx, mask_ctx, test, mask_test, inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularData,
    lambda d: ((d.feature_groups, d.f), None),
    lambda _aux, children: TabularData(*children),
)


@dataclass(frozen=True, eq=False)
class TabularBatch(MetaLearningBatch):
    ctx: FrozenDict[str, jax.Array]
    mask_ctx: Optional[jax.Array]
    test: FrozenDict[str, jax.Array]
    mask_test: Optional[jax.Array] = None
    inv_permute_idx: Optional[jax.Array] = None

    # NOTE: these methods are defined so that when **TabularBatch() is used, the
    # `ctx` and `test` attributes are expanded into their groups, e.g.
    # {'x_ctx': <array>, 's_ctx': <array>, 'f_ctx': <array>, ...} and
    # {'x_test': <array>, 's_test': <array>, 'f_test': <array>, ...}
    def __iter__(self):
        yield from self.ctx
        yield "mask_ctx"
        yield from self.test
        yield "mask_test"
        yield "inv_permute_idx"

    def __len__(self):
        return len(self.ctx) + len(self.test) + 3

    def __getitem__(self, key):
        if key in ["mask_ctx", "mask_test", "inv_permute_idx"]:
            return getattr(self, key)
        return self.ctx[key] if "ctx" in key else self.test[key]


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TabularBatch,
    lambda d: (
        (
            d.ctx,
            d.mask_ctx,
            d.test,
            d.mask_test,
            d.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: TabularBatch(*children),
)
