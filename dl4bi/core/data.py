from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random, vmap


@dataclass(frozen=True)
class BatchElement:
    x_ctx: Optional[jax.Array] = None  # [L_ctx, D_x]
    s_ctx: Optional[jax.Array] = None  # [L_ctx, D_s]
    f_ctx: Optional[jax.Array] = None  # [L_ctx, D_f]
    valid_lens_ctx: Optional[jax.Array] = None  # [1]
    x_test: Optional[jax.Array] = None  # [L_test, D_x]
    s_test: Optional[jax.Array] = None  # [L_test, D_s]
    f_test: Optional[jax.Array] = None  # [L_test, D_f]
    valid_lens_test: Optional[jax.Array] = None  # [1]
    inv_permute_idx: Optional[jax.Array] = None  # [T, L] or [L]
    time_step_idx: Optional[jax.Array] = None  # [T]

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None


# register Batch with jax so it can be used in compiled functions
jax.tree_util.register_pytree_node(
    BatchElement,
    lambda b: (
        (
            b.x_ctx,
            b.s_ctx,
            b.f_ctx,
            b.valid_lens_ctx,
            b.x_test,
            b.s_test,
            b.f_test,
            b.valid_lens_test,
            b.inv_permute_idx,
            b.time_step_idx,
        ),
        None,
    ),
    lambda _aux, children: BatchElement(*children),
)


@dataclass(frozen=True)
class Batch:
    x_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_x]
    s_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_s]
    f_ctx: Optional[jax.Array] = None  # [B, L_ctx, D_f]
    valid_lens_ctx: Optional[jax.Array] = None  # [B]
    x_test: Optional[jax.Array] = None  # [B, L_test, D_x]
    s_test: Optional[jax.Array] = None  # [B, L_test, D_s]
    f_test: Optional[jax.Array] = None  # [B, L_test, D_f]
    valid_lens_test: Optional[jax.Array] = None  # [B]
    inv_permute_idx: Optional[jax.Array] = None  # [B, T, L] or [B, L] or [L]
    time_step_idx: Optional[jax.Array] = None  # [B, T]

    def update(self, **kwargs):
        """Returns a new batch with updated attributes."""
        return replace(self, **kwargs)

    def __iter__(self):
        """Allows you to use **batch to expand as kwargs."""
        yield from asdict(self).items()

    def __getitem__(self, i: int):
        d = {}
        for k, v in iter(self):
            if isinstance(v, jax.Array):
                if k.endswith(("ctx", "test")):
                    v = v[i]  # [B] or [B, L, D]
                elif k == "inv_permute_idx":
                    # inv_permute_idx shape meaning:
                    # [L]: A single permutation for all batch elements
                    # [B, L]: A separate permutation for each batch element
                    # [B, T, L]: A separate permutation for each batch element and timestep
                    if v.ndim > 1:
                        v = v[i]
                elif k == "time_start_idx":
                    v = v[i]
            d[k] = v
        return BatchElement(**d)

    def __len__(self):
        for field in fields(self):
            if field.name.endswith(("ctx", "test")):
                x = getattr(self, field.name)
                if isinstance(x, jax.Array):
                    return x.shape[0]
        return None


# register Batch with jax so it can be used in compiled functions
jax.tree_util.register_pytree_node(
    Batch,
    lambda b: (
        (
            b.x_ctx,
            b.s_ctx,
            b.f_ctx,
            b.valid_lens_ctx,
            b.x_test,
            b.s_test,
            b.f_test,
            b.valid_lens_test,
            b.inv_permute_idx,
            b.time_step_idx,
        ),
        None,
    ),
    lambda _aux, children: Batch(*children),
)


@dataclass(frozen=True)
class SpatialData:
    x: Optional[jax.Array]  # [B, [S]+, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]

    def to_batch(
        self,
        rng: jax.Array,
        min_ctx: int,
        max_ctx: Optional[int],
        num_test: Optional[int],
        independent: bool = False,
        test_includes_ctx: bool = True,
    ):
        return spatial_data_to_batch(
            rng,
            self.x,
            self.s,
            self.f,
            min_ctx,
            max_ctx,
            num_test,
            independent,
            test_includes_ctx,
        )


# register SpatialData with jax so it can be used in compiled functions
jax.tree_util.register_pytree_node(
    SpatialData,
    lambda d: ((d.x, d.s, d.f), None),
    lambda _aux, children: SpatialData(*children),
)


@partial(
    jit,
    static_argnames=(
        "min_ctx",
        "max_ctx",
        "num_test",
        "independent",
        "test_includes_ctx",
    ),
)
def spatial_data_to_batch(
    rng: jax.Array,
    x: Optional[jax.Array],  # [B, [S]+, D_x] or None
    s: jax.Array,  # [B, [S]+]
    f: jax.Array,  # [B, [S]+, D_f]
    min_ctx: int = 1,
    max_ctx: Optional[int] = None,
    num_test: Optional[int] = None,
    independent: bool = False,
    test_includes_ctx: bool = True,
):
    B = s.shape[0]
    has_x = x is not None
    flatten_spatial_dims = lambda v: v.reshape(B, -1, v.shape[-1])
    x = flatten_spatial_dims(x) if has_x else None
    s = flatten_spatial_dims(s)
    f = flatten_spatial_dims(f)
    L = s.shape[1]
    num_test = num_test or L
    max_ctx = max_ctx or L - num_test
    rng_valid, rng = random.split(rng)
    valid_lens_ctx = random.randint(rng_valid, (B,), min_ctx, max_ctx)
    valid_lens_test = jnp.repeat(num_test, B)
    if independent:  # each element of the batch is permunted independently
        rngs = random.split(rng, B)
        if has_x:
            output = vmap(_spatial_helper, in_axes=(0, 0, 0, 0, None, None, None))(
                rngs,
                # add an extra dim to use the helper
                x[:, None],
                s[:, None],
                f[:, None],
                max_ctx,
                num_test,
                test_includes_ctx,
            )
        else:
            output = vmap(_spatial_helper, in_axes=(0, None, 0, 0, None, None, None))(
                rngs,
                None,
                # add an extra dim to use the helper
                s[:, None],
                f[:, None],
                max_ctx,
                num_test,
                test_includes_ctx,
            )
        # remove the extra dim
        x_ctx, s_ctx, f_ctx, x_test, s_test, f_test, inv_permute_idx = output
        x_ctx, x_test = (x_ctx[:, 0], x_test[:, 0]) if has_x else (None, None)
        s_ctx, s_test = s_ctx[:, 0], s_test[:, 0]
        f_ctx, f_test = f_ctx[:, 0], f_test[:, 0]
    else:  # use same permutation for all elements in the batch
        x_ctx, s_ctx, f_ctx, x_test, s_test, f_test, inv_permute_idx = _spatial_helper(
            rng,
            x,
            s,
            f,
            max_ctx,
            num_test,
            test_includes_ctx,
        )
    return Batch(
        x_ctx,
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        x_test,
        s_test,
        f_test,
        valid_lens_test,
        inv_permute_idx,
    )


@partial(jit, static_argnames=("max_ctx", "num_test", "test_includes_ctx"))
def _spatial_helper(
    rng: jax.Array,
    x: Optional[jax.Array],  # [B, L, D_x] or None
    s: jax.Array,  # [B, L, D_s]
    f: jax.Array,  # [B, L, D_f]
    max_ctx: int,
    num_test: int,
    test_includes_ctx: bool = True,
):
    L = s.shape[1]
    has_x = x is not None
    permute_idx = random.choice(rng, L, (L,), replace=False)
    inv_permute_idx = jnp.argsort(permute_idx)
    x_permuted = x[:, permute_idx] if has_x else None
    s_permuted = s[:, permute_idx]
    f_permuted = f[:, permute_idx]
    x_ctx = x_permuted[:, :max_ctx] if has_x else None
    s_ctx = s_permuted[:, :max_ctx]
    f_ctx = f_permuted[:, :max_ctx]
    if test_includes_ctx:
        x_test = x_permuted[:, :num_test] if has_x else None
        s_test = s_permuted[:, :num_test]
        f_test = f_permuted[:, :num_test]
    else:
        x_test = x_permuted[:, max_ctx : max_ctx + num_test] if has_x else None
        s_test = s_permuted[:, max_ctx : max_ctx + num_test]
        f_test = f_permuted[:, max_ctx : max_ctx + num_test]
    return x_ctx, s_ctx, f_ctx, x_test, s_test, f_test, inv_permute_idx


# @partial(
#     jit,
#     static_argnames=(
#         "s_min_ctx",
#         "s_max_ctx",
#         "t_indep_s_ctx",
#         "t_fixed_interval",
#         "t_target_is_last",
#         "batch_size",
#     ),
# )
# def spatiotemporal_data_to_batch(
#     rng: jax.Array,
#     x: Optional[jax.Array],  # [T, [S]+, D_x] or [[S]+, D_x] or None
#     s: jax.Array,  # [T, [S]+] or [S]+
#     t: jax.Array,  # [T]
#     f: jax.Array,  # [T, [S]+, D_f]
#     s_min_ctx: Optional[int] = None,
#     s_max_ctx: Optional[int] = None,
#     t_num_ctx: int = 4,
#     t_indep_s_ctx: bool = False,
#     t_fixed_interval: Optional[int] = None,
#     t_target_is_last: bool = True,
#     batch_size: int = 4,
# ):
#     B = batch_size
#     rng_ts, rng = random.split(rng)
#     if t_target_is_last:  # last time step is the target time step
#         rng_lasts, rng_ts = random.split(rng)
#         t_lasts = random.randint(rng_lasts, (B,), t.min() + t_num_ctx, t.max())
#         ts = []
#         for t_last in t_lasts:
#             rng_t, rng_ts = random.split(rng_ts)
#             ts += [random.randint(rng_t, ())]

#     pass
