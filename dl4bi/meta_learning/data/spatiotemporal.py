from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import jit, random, vmap
from jax.scipy.stats import norm

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    flatten_spatial,
    inv_permute_L_in_BLD,
    permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True, eq=False)
class SpatiotemporalData(MetaLearningData):
    """
    .. warning::
        This class assumes that time, `t`, is ordered and ascending.
    """

    x: Optional[jax.Array]  # [T, [S]+, D_x] or [B, T, D_x] or [B, D_x] or None
    s: jax.Array  # [T, [S]+, D_s]
    t: jax.Array  # [T]
    f: jax.Array  # [T, [S]+, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_t: int,
        random_t: bool,
        num_ctx_min_per_t: int,
        num_ctx_max_per_t: int,
        independent_t_masks: bool,
        num_test: int,
        forecast: bool,
        batch_size: int = 4,
    ):
        """
        Args:
            rng: A PRNG.
            num_t: Number of time steps in each batch element.
            random_t: When `True` selects random time steps and when `False`
                selects a a sequential set of `num_t` time steps for each
                batch element.
            num_ctx_min_per_t: Minumum mumber of observable spatial context
                points per time step.
            num_ctx_max_per_t: Maximum mumber of observable spatial context
                points per time step.
            independent_t_masks: When `True`, the spatial context points
                for each time step are chosen independently. When `False`,
                the observed spatial locations are the same across all
                time steps.
            num_test: Number of test points in the test time step.
            forecast: If `True`, the predicted test points are part of the last
                time step in the batch element. Otherwise, this is assumed to
                be an interpolation task and test points come from the median
                time step.
            batch_size: The batch size.
        """
        return _batch(
            rng,
            self.x,
            self.s,
            self.t,
            self.f,
            num_t,
            random_t,
            num_ctx_min_per_t,
            num_ctx_max_per_t,
            independent_t_masks,
            num_test,
            forecast,
            batch_size,
        )


@partial(
    jit,
    static_argnames=(
        "num_t",
        "random_t",
        "num_ctx_min_per_t",
        "num_ctx_max_per_t",
        "independent_t_masks",
        "num_test",
        "forecast",
        "batch_size",
    ),
)
def _batch(
    rng: jax.Array,
    x: Optional[jax.Array],
    s: jax.Array,
    t: jax.Array,
    f: jax.Array,
    num_t: int,
    random_t: bool,
    num_ctx_min_per_t: int,
    num_ctx_max_per_t: int,
    independent_t_masks: bool,
    num_test: int,
    forecast: bool,
    batch_size: int = 4,
):
    B, T, T_b = batch_size, t.shape[1], num_t
    rng_t, rng_p, rng_b = random.split(rng, 3)
    has_x = x is not None
    broadcast_x = x.ndim in (2, 3) if has_x else None
    ts_test, ts_ctx = _select_ts(rng_t, random_t, forecast, B, T, T_b)
    s, f = map(flatten_spatial, [s, f])
    tpls = map(lambda v: (v[ts_ctx], v[ts_test]), [s, t, f])
    (s_ctx, s_test), (t_ctx, t_test), (f_ctx, f_test) = tpls
    t_ctx = jnp.broadcast_to(t_ctx[:, :, None, None], (f_ctx.shape[:-1], 1))
    t_test = jnp.broadcast_to(t_test[:, :, None, None], (f_test.shape[:-1], 1))
    if has_x and not broadcast_x:  # x: [T, [S]+, D_x]
        x = flatten_spatial(x)  # [T, L_s, D_x]
        x_ctx, x_test = x[ts_ctx], x[ts_test]
    elif broadcast_x:
        D_x = x.shape[-1]
        ctx_shape, test_shape = (*f_ctx.shape[:-1], D_x), (*f_test.shape[:-1], D_x)
        # if ndim = 3, broadcast over space, otherwise broadcast over time and space
        x = x[:, :, None, :] if x.ndim == 3 else x[:, None, None, :]
        x_ctx = jnp.broadcast_to(x, ctx_shape)
        x_test = jnp.broadcast_to(x, test_shape)
    # *_ctx: [B, T_b-1, L_s, D_*], *_test: [B, 1, L_s, D_*]
    # TODO(danj): permute L_s, select valid_lens and truncate
    # vmap(batch_BLD)

    # s, f = flatten_spatial(s), flatten_spatial(f)
    # batch_args = (num_ctx_min_per_t, num_ctx_max_per_t, num_test)
    # if broadcast_x or not has_x:
    #     x_ctx = x_test = x
    #     s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [s, f])
    #     args = batch_BLD(rng_b, [s, f], *batch_args)
    #     s_ctx, f_ctx, v_ctx, s_test, f_test, *rest = args
    #     if broadcast_x:
    #         x_ctx = jnp.repeat(x_ctx[:, None], num_ctx_max, axis=1)
    #         x_test = jnp.repeat(x_test[:, None], num_test, axis=1)
    #     args = (x_ctx, s_ctx, f_ctx, v_ctx, x_test, s_test, f_test, *rest)
    # else:
    #     x = flatten_spatial(x)
    #     x, s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [x, s, f])
    #     args = batch_BLD(rng_b, [x, s, f], *batch_args)
    # return SpatiotemporalBatch(*args, inv_permute_idx=inv_permute_idx, s_shape=s.shape)


def _select_ts(
    rng_t: jax.Array,
    random_t: bool,
    forecast: bool,
    B: int,
    T: int,
    T_b: int,
):
    if random_t:
        rng_ts = random.split(rng_t, B)
        ts = vmap(lambda rng: random.choice(rng, T, (T_b,)))(rng_ts)
        ts = jnp.sort(ts, axis=1)  # [B, T_b]
    else:
        last_ts = random.randint(rng_t, (T_b,), T_b, T)
        prev_ts = jnp.arange(T_b - 1, -1, -1)
        ts = last_ts[:, None] - prev_ts  # [B, T_b]
    test_i = -1 if forecast else T_b // 2 + 1  # last time step or median
    # get indices of time steps for batches
    ts_ctx = jnp.concat([ts[:, :test_i], ts[:, test_i + 1 :]], axis=1)
    ts_test = ts[:, [test_i]]
    return ts_ctx, ts_test


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    SpatiotemporalData,
    lambda d: ((d.x, d.s, d.t, d.f), None),
    lambda _aux, children: SpatiotemporalData(*children),
)


@dataclass(frozen=True, eq=False)
class SpatiotemporalBatch(MetaLearningBatch):
    x_ctx: Optional[jax.Array]  # [B, L_ctx, D_x] or None
    s_ctx: jax.Array  # [B, L_ctx, D_s]
    t_ctx: jax.Array  # [B, L_ctx, 1]
    f_ctx: jax.Array  # [B, L_ctx, D_f]
    valid_lens_ctx: jax.Array  # [B]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    s_test: jax.Array  # [B, L_test, D_s]
    t_test: jax.Array  # [B, L_test, 1]
    f_test: jax.Array  # [B, L_test, D_f]
    valid_lens_test: jax.Array  # [B]
    inv_permute_idx: jax.Array  # [L]
    s_shape: tuple

    def plot_1d(
        self,
        f_pred: jax.Array,  # [B, [K]?, L_test, 1]
        f_std: jax.Array,  # [B, [K]?, L_test, 1]
        hdi_prob: float = 0.95,
    ):
        (B, L_test), L = self.f_test.shape[:2], self.inv_permute_idx.shape[0]
        K = 1 if f_pred.ndim != 4 else f_pred.shape[1]  # bootstrapped K
        f_pred = f_pred.reshape(B * K, L_test, -1)
        f_std = f_std.reshape(B * K, L_test, -1)
        arrays = [self.s_test, self.f_test, f_pred, f_std]
        arrays = unbatch_BLD(arrays, L)
        arrays = inv_permute_L_in_BLD(arrays, self.inv_permute_idx)
        s_test, f_test, f_pred, f_std = map(lambda v: v[..., 0], arrays)
        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        f_lower, f_upper = f_pred - z_score * f_std, f_pred + z_score * f_std
        _, axs = plt.subplots(B, 1, figsize=(8, B * 4))
        for i in range(B):
            if i == 0:
                axs[i].set_title("Spatial Posterior Predictive")
            elif i == B - 1:
                axs[i].set_xlabel("s")
            axs[i].set_ylabel(f"Sample {i+1}", rotation=90)
            axs[i].scatter(self.s_ctx[i, :, 0], self.f_ctx[i, :, 0], color="black")
            axs[i].plot(s_test[i], f_test[i], color="black")
            for j in range(K):
                axs[i].plot(s_test[i], f_pred[i], color="steelblue")
                axs[i].fill_between(
                    s_test[i],
                    f_lower[i],
                    f_upper[i],
                    alpha=0.4 / K,
                    color="steelblue",
                    interpolate=True,
                )
        plt.tight_layout()
        return plt.gcf()

    def plot_2d(
        self,
        f_pred: jax.Array,  # [B, [K]?, L_test, D_f]
        f_std: jax.Array,  # [B, [K]?, L_test, D_f]
        cmap=mpl.colormaps.get_cmap("grey"),
        cmap_std=mpl.colormaps.get_cmap("Spectral_r"),
        norm=None,
        norm_std=None,
        remap_colors: Callable = lambda x: x,
    ):
        B, L = self.f_test.shape[0], self.inv_permute_idx.shape[0]
        reshape = jit(lambda v: v.reshape(*self.s_shape[:-1], v.shape[-1]))
        if f_pred.ndim == 4:  # [B, K, L, D] boostrapped
            f_pred = f_pred.mean(axis=1)
            f_std = f_std.mean(axis=1)
        if f_std.shape[-1] > 1:  # e.g. uncertainty per RGB channel
            f_std = f_std.mean(axis=-1)
        arrays = unbatch_BLD([self.f_ctx, self.f_test, f_pred, f_std], L)
        arrays = inv_permute_L_in_BLD(arrays, self.inv_permute_idx)
        arrays = map(reshape, arrays)
        f_ctx, f_test, f_pred, f_std = map(remap_colors, arrays)
        _, axs = plt.subplots(B, 4, figsize=(20, B * 5))
        for i in range(B):
            if i == 0:
                axs[i, 0].set_title("Task")
                axs[i, 1].set_title("Uncertainty")
                axs[i, 2].set_title("Prediction")
                axs[i, 3].set_title("Ground Truth")
            axs[i, 0].set_ylabel(f"Sample {i}")
            axs[i, 0].imshow(f_ctx, cmap=cmap, norm=norm, interpolation="none")
            axs[i, 1].imshow(f_std, cmap=cmap_std, norm=norm_std, interpolation="none")
            axs[i, 2].imshow(f_pred, cmap=cmap, norm=norm, interpolation="none")
            axs[i, 3].imshow(f_test, cmap=cmap, norm=norm, interpolation="none")
        plt.tight_layout()
        return plt.gcf()


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    SpatiotemporalBatch,
    lambda d: (
        (
            d.x_ctx,
            d.s_ctx,
            d.t_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.s_test,
            d.t_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.s_shape,),
    ),
    lambda aux, children: SpatiotemporalBatch(*children, *aux),
)


# class SpatiotemporalTask(Enum):
#     Forecast = "forecast"
#     Interpolate = "interpolate"


# @dataclass(frozen=True)
# class SpatiotemporalData:
#     """A `SpatiotemporalData` container.

#     This class is intended for datasets with a spatial dimension, `s`, a
#     temporal dimension, `t`, optional fixed effects per time step per location, `x`,
#     and the functional output associated with each time step and location.

#     .. note::
#         Unlike the other `Data` classes, the batch dimension here is the time, `T`.

#     .. warning::
#         This container assumes that that timesteps are ordered from least to
#         most recent.
#     """

#     x: Optional[jax.Array]  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
#     s: jax.Array  # [T, [S]+, D_s]
#     t: jax.Array  # [T, [S]+, 1] or [T, 1, 1]
#     f: jax.Array  # [T, [S]+, D_f]

#     def to_batch(
#         self,
#         rng: jax.Array,
#         task: SpatiotemporalTask,
#         fixed_t_step: int,
#         num_t_per_element: int,
#         num_ctx_min_per_t: int,
#         num_ctx_max_per_t: int,
#         independent: bool,
#         num_test: int,
#         batch_size: int,
#         include_inv_permute_idx: bool = False,
#     ):
#         """Creates a `Batch` from this `SpatiotemporalData`.

#         Args:
#             rng: A PRNG.
#             task: A `SpatiotemporalTask` type, e.g. interpolate or forecast.
#             fixed_t_step: If greater than 0, use timesteps separated by a fixed
#                 step size along the leading axis, `T`, e.g. a `fixed_t_step=2`
#                 would yield indices 0, 2, 4, 6, ... along the `T` axis. Note
#                 that this likely only makes sense when the elements in the
#                 leading axis, `T`, are separated by a fixed time interval, e.g.
#                 t[0]=1.0, t[1]=2.0, t[2]=3.0, etc. If the time steps along
#                 the leading axis are real numbers with irregular gaps, e.g.
#                 t[0]=1.23, t[1]=5.7, t[3]=20.8, etc, this will likely have an
#                 unintended effect. In summary, `fixed_t_step` works with fixed
#                 indices along `T`, not with the actual times they represent.
#             num_t_per_element: Number of time steps to include in each batch
#                 element.
#             num_ctx_min_per_t: Minimum number of context points per time step.
#             num_ctx_max_per_t: Maximum number of context points per time step.
#             independent: Whether the subset of context points for each time step
#                 should be selected independently.
#             num_test: Number of test points for target time step.
#             batch_size: Number of elements in the generated batch.
#             include_inv_permute_idx: Whether to include the `inv_permute_idx`,
#                 which enables easily mapping context points back to their
#                 original times and locations in the dataset. This is useful for
#                 callbacks and plotting functions.
#         Returns:
#             A `Batch`.
#         """
#         return _spatiotemporal_data_to_batch(
#             rng,
#             self.x,
#             self.s,
#             self.t,
#             self.f,
#             task,
#             fixed_t_step,
#             num_t_per_element,
#             num_ctx_min_per_t,
#             num_ctx_max_per_t,
#             independent,
#             num_test,
#             batch_size,
#             include_inv_permute_idx,
#         )


# # register with jax to use in compiled functions
# jax.tree_util.register_pytree_node(
#     SpatiotemporalData,
#     lambda d: ((d.x, d.s, d.t, d.f), None),
#     lambda _aux, children: SpatiotemporalData(*children),
# )


# @partial(
#     jit,
#     static_argnames=(
#         "task",
#         "fixed_t_step",
#         "num_t_per_element",
#         "num_ctx_min_per_t",
#         "num_ctx_max_per_t",
#         "independent",
#         "num_test",
#         "batch_size",
#         "include_inv_permute_idx",
#     ),
# )
# def _spatiotemporal_data_to_batch(
#     rng: jax.Array,
#     x: Optional[jax.Array],  # [T, [S]+, D_x] or [T, 1, D_x] or [1, 1, D_x] or None
#     s: jax.Array,  # [T, [S]+, D_s]
#     t: jax.Array,  # [T, [S]+, 1] or [T, 1, 1]
#     f: jax.Array,  # [T, [S]+, D_f]
#     task: SpatiotemporalTask,
#     fixed_t_step: int,
#     num_t_per_element: int,
#     num_ctx_min_per_t: int,
#     num_ctx_max_per_t: int,
#     independent: bool,
#     num_test: int,
#     batch_size: int = 4,
#     include_inv_permute_idx: bool = False,
# ):
#     B, T, T_e = batch_size, s.shape[0], num_t_per_element
#     has_x = x is not None
#     flatten_spatial_dims = lambda v: v.reshape(T, -1, v.shape[-1])
#     x = flatten_spatial_dims(x) if has_x else None
#     s = flatten_spatial_dims(s)
#     t = flatten_spatial_dims(t)
#     f = flatten_spatial_dims(f)
#     L = s.shape[1]
#     num_test = num_test or L
#     x = jnp.broadcast_to(x, (T, L, x.shape[-1])) if has_x else None
#     t = jnp.broadcast_to(t, (T, L, 1))
#     rng_v, rng_t, rng = random.split(rng, 3)
#     valid_lens_ctx = random.randint(
#         rng_v, (B, T_e - 1), num_ctx_min_per_t, num_ctx_max_per_t
#     )
#     valid_lens_test = jnp.repeat(num_test, T)
#     if fixed_t_step < 0:  # use random time steps
#         rng_ts = random.split(rng_t, B)
#         ts = vmap(lambda rng: random.choice(rng, T, (T_e,), replace=False))(rng_ts)
#         ts = jnp.sort(ts, axis=1)
#     else:
#         rng_ts = random.split(rng_t, B)
#         choices = jnp.arange(fixed_t_step * (num_t_per_element - 1), T)
#         last_ts = random.choice(rng, choices, (B,), replace=False)
#         prev_ts = fixed_t_step * jnp.arange(num_t_per_element - 1, -1, -1)
#         ts = last_ts[:, None] - prev_ts
#     # last step if forecast, else median for interpolate
#     test_i = -1 if task is SpatiotemporalTask.Forecast else num_t_per_element // 2 + 1
#     # get indices of time steps for batches
#     t_test_i = ts[:, [test_i]]
#     t_ctx_i = jnp.concat([ts[:, :test_i], ts[:, test_i + 1 :]], axis=1)
#     # shapes: *_ctx: [B, T_e-1, [S]+, D_*], *_test: [B, 1, [S]+, D_*]
#     x_ctx, x_test = (x[t_ctx_i], x[t_test_i]) if has_x else (None, None)
#     s_ctx, s_test = s[t_ctx_i], s[t_test_i]
#     t_ctx, t_test = t[t_ctx_i], t[t_test_i]
#     f_ctx, f_test = f[t_ctx_i], f[t_test_i]
#     rngs = random.split(rng, T_e) if independent else jnp.repeat(rng, T_e)
#     # TODO:
#     # permute each T_e-1 ctx
#     # permute and pack test locs
#     # pack the context into a single array
#     x_ctx, x_test, _ = vbatch(rngs, x) if has_x else (None, None, None)
#     s_ctx, s_test, _ = vbatch(rngs, s)
#     t_ctx, t_test, _ = vbatch(rngs, t)
#     f_ctx, f_test, inv_permute_idx = vbatch(rngs, f)
#     return Batch(
#         x_ctx,
#         s_ctx,
#         t_ctx,
#         f_ctx,
#         valid_lens_ctx,
#         x_test,
#         s_test,
#         t_test,
#         f_test,
#         valid_lens_test,
#         # no need to store this if its not used
#         inv_permute_idx if include_inv_permute_idx else None,
#     )
