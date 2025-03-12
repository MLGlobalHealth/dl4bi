from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import jit, random, vmap

from ...core.utils import mask_from_valid_lens, nan_pad
from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    _permute_idx,
    _vpermute_idx,
    inv_permute_L_in_BLD,
    unbatch_BLD,
)


@dataclass(frozen=True, eq=False)
class SpatiotemporalData(MetaLearningData):
    """
    .. warning::
        This class assumes that time, `t`, is ordered and ascending.
    """

    x: Optional[jax.Array]  # [T, [S]+, D_x] or [T, D_x] or [D_x] None
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
    B, T, T_b = batch_size, t.shape[0], num_t
    rng_t, rng_p, rng_b, rng_v = random.split(rng, 4)
    s_dims = s.shape[1:-1]
    has_x = x is not None
    full_x = has_x and x.shape[:-1] == s.shape[:-1]
    broadcast_x = x.ndim in (1, 2) if has_x else None
    ts_ctx, ts_test = _select_ts(rng_t, random_t, forecast, B, T, T_b)
    S_to_L = jit(lambda v: v.reshape(v.shape[0], -1, v.shape[-1]))
    s, f = map(S_to_L, [s, f])
    splits = map(lambda v: (v[ts_ctx], v[ts_test]), [s, t, f])
    (s_ctx, s_test), (t_ctx, t_test), (f_ctx, f_test) = splits  # [B, T_b, L_s, D]
    t_ctx = jnp.broadcast_to(t_ctx[:, :, None, None], (*f_ctx.shape[:-1], 1))
    t_test = jnp.broadcast_to(t_test[:, :, None, None], (*f_test.shape[:-1], 1))
    if full_x:  # x: [T, [S]+, D_x]
        x = S_to_L(x)  # [T, L_s, D_x]
        x_ctx, x_test = x[ts_ctx], x[ts_test]
    elif broadcast_x:
        D_x = x.shape[-1]
        has_t = x.ndim == 2
        x = x if has_t else x[None, :]  # [T, D_x] or [1, D_x]
        x_ctx, x_test = (x[ts_ctx], x[ts_test]) if has_t else (x[:, None], x[:, None])
        ctx_shape, test_shape = (*f_ctx.shape[:-1], D_x), (*f_test.shape[:-1], D_x)
        # expand spatial dims, i.e. [B, T_b, D_x] -> [B, T_b, [S]+, D_x]
        while len(x_ctx.shape) < len(ctx_shape):
            x_ctx = x_ctx[:, :, None]
            x_test = x_test[:, :, None]
        x_ctx = jnp.broadcast_to(x_ctx, ctx_shape)
        x_test = jnp.broadcast_to(x_test, test_shape)
    # *_ctx: [B, T_b-1, L_s, D_*], *_test: [B, 1, L_s, D_*]
    mask_ctx, mask_test = _build_masks(
        rng_v,
        num_ctx_min_per_t,
        num_ctx_max_per_t,
        num_test,
        independent_t_masks,
        B,
        T_b,
    )
    s_ctx, s_test, _ = _permute_Ls(rng_p, s_ctx, s_test, independent_t_masks)
    t_ctx, t_test, _ = _permute_Ls(rng_p, t_ctx, t_test, independent_t_masks)
    f_ctx, f_test, inv_permute_idx = _permute_Ls(
        rng_p, f_ctx, f_test, independent_t_masks
    )
    flatten_ts = jit(lambda v: v.reshape(v.shape[0], -1, v.shape[-1]))
    if has_x:
        x_ctx, x_test, _ = _permute_Ls(rng_p, x_ctx, x_test, independent_t_masks)
        ctxs, tests = [x_ctx, s_ctx, t_ctx, f_ctx], [x_test, s_test, t_test, f_test]
        ctxs = [v[:, :, :num_ctx_max_per_t] for v in ctxs]
        tests = [v[:, :, :num_test] for v in tests]
        ctxs = list(map(flatten_ts, ctxs + [mask_ctx[..., None]]))
        tests = list(map(flatten_ts, tests + [mask_test[..., None]]))
    else:
        ctxs, tests = [s_ctx, t_ctx, f_ctx], [s_test, t_test, f_test]
        ctxs = [v[:, :, :num_ctx_max_per_t] for v in ctxs]
        tests = [v[:, :, :num_test] for v in tests]
        ctxs = [None] + list(map(flatten_ts, ctxs + [mask_ctx[..., None]]))
        tests = [None] + list(map(flatten_ts, tests + [mask_test[..., None]]))
    ctxs[-1] = ctxs[-1][..., 0]  # remove extra mask dim
    tests[-1] = tests[-1][..., 0]  # remove extra mask dim
    return SpatiotemporalBatch(
        *ctxs,
        *tests,
        inv_permute_idx=inv_permute_idx,
        s_dims=s_dims,
        forecast=forecast,
    )


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
        ts = vmap(lambda rng: random.choice(rng, T, (T_b,), replace=False))(rng_ts)
        ts = jnp.sort(ts, axis=1)  # [B, T_b]
    else:  # sequential times
        last_ts = random.randint(rng_t, (B,), T_b, T)
        delta_ts = jnp.arange(T_b - 1, -1, -1)
        ts = last_ts[:, None] - delta_ts  # [B, T_b]
    test_i = T_b - 1 if forecast else T_b // 2  # last time step or median
    # get indices of time steps for batches
    ts_ctx = jnp.concat([ts[:, :test_i], ts[:, test_i + 1 :]], axis=1)
    ts_test = ts[:, [test_i]]
    return ts_ctx, ts_test


def _build_masks(
    rng: jax.Array,
    num_ctx_min_per_t: int,
    num_ctx_max_per_t: int,
    num_test: int,
    independent_t_masks: bool,
    B: int,
    T_b: int,
):
    Nc_min, Nc_max = num_ctx_min_per_t, num_ctx_max_per_t
    if independent_t_masks:
        valid_lens_ctx_per_t = random.randint(rng, (T_b - 1,), Nc_min, Nc_max)
    else:
        valid_lens_ctx_per_t = random.randint(rng, (1,), Nc_min, Nc_max)
        valid_lens_ctx_per_t = jnp.repeat(valid_lens_ctx_per_t, T_b - 1)
    mask_ctx = mask_from_valid_lens(Nc_max, valid_lens_ctx_per_t)
    mask_ctx = jnp.repeat(mask_ctx[None, ...], B, axis=0)
    valid_lens_test = jnp.repeat(num_test, B)
    mask_test = mask_from_valid_lens(num_test, valid_lens_test)[:, None]
    return mask_ctx, mask_test  # [B, {T_b-1,1}, N]


def _permute_Ls(
    rng: jax.Array,
    v_ctx: jax.Array,
    v_test: jax.Array,
    independent_t_masks: bool = True,
):
    B, T_b_minus_1, L_s, _ = v_ctx.shape
    T_b = T_b_minus_1 + 1
    v = jnp.concat([v_ctx, v_test], axis=1)  # [B, T_b, L_s, D]
    if independent_t_masks:
        rng_ps = random.split(rng, T_b)
        permute_idxs = _vpermute_idx(rng_ps, L_s)
    else:
        permute_idx = _permute_idx(rng, L_s)
        permute_idxs = jnp.repeat(permute_idx[None, ...], T_b, axis=0)
    inv_permute_idxs = jnp.argsort(permute_idxs, axis=1)
    func = vmap(vmap(lambda v, idx: v[idx]), in_axes=(0, None))
    v_perm = func(v, permute_idxs)
    return v_perm[:, :-1], v_perm[:, [-1]], inv_permute_idxs


def _inv_permute_Ls(v_ctx: jax.Array, v_test: jax.Array, inv_permute_idx: jax.Array):
    B = v_ctx.shape[0]
    v = jnp.concat([v_ctx, v_test], axis=1)
    vs = []
    for i in range(B):
        vs += [inv_permute_L_in_BLD([v[i]], inv_permute_idx)[0]]
    v = jnp.array(vs)
    return v[:, :-1], v[:, [-1]]


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
    mask_ctx: jax.Array  # [B, L_ctx]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    s_test: jax.Array  # [B, L_test, D_s]
    t_test: jax.Array  # [B, L_test, 1]
    f_test: jax.Array  # [B, L_test, D_f]
    mask_test: jax.Array  # [B, L_test]
    inv_permute_idx: jax.Array  # [T_b, L]
    s_dims: tuple
    forecast: bool

    def plot_2d(
        self,
        f_pred: jax.Array,  # [B, L_test, D_f]
        f_std: jax.Array,  # [B, L_test, D_f]
        cmap=mpl.colormaps.get_cmap("grey"),
        cmap_std=mpl.colormaps.get_cmap("Spectral_r"),
        norm=None,
        norm_std=None,
        remap_colors: Callable = lambda x: x,
    ):
        (T_b, L), (B, _, D_f) = self.inv_permute_idx.shape, f_pred.shape
        # fill in masked values with nan
        f_ctx = jnp.where(self.mask_ctx[..., None], self.f_ctx, jnp.nan)
        f_test = self.f_test
        if self.mask_test is not None:
            f_test = jnp.where(self.mask_test[..., None], self.f_test, jnp.nan)
        # reintroduce timestep and nan pad each time step to full size
        f_ctx = f_ctx.reshape(B, T_b - 1, -1, D_f)
        f_ctx = nan_pad(f_ctx, axis=2, L=L)
        f_test, f_pred, f_std = unbatch_BLD([f_test, f_pred, f_std], L)
        f_test, f_pred, f_std = map(lambda v: v[:, None], [f_test, f_pred, f_std])
        # invert permutation of the flattened spatial dim, L, by time step
        _, f_test = _inv_permute_Ls(f_ctx, f_test, self.inv_permute_idx)
        _, f_pred = _inv_permute_Ls(f_ctx, f_pred, self.inv_permute_idx)
        f_ctx, f_std = _inv_permute_Ls(f_ctx, f_std, self.inv_permute_idx)
        # reshape to original spatial dims
        reshape_s = jit(lambda v: v.reshape(*v.shape[:2], *self.s_dims, v.shape[-1]))
        f_ctx, f_test, f_pred, f_std = map(reshape_s, [f_ctx, f_test, f_pred, f_std])
        f_ctx, f_test, f_pred = map(remap_colors, [f_ctx, f_test, f_pred])
        if f_std.shape[-1] > 1:  # e.g. uncertainty per RGB channel
            f_std = f_std.mean(axis=-1, keepdims=True)
        _, axs = plt.subplots(B, T_b + 2, figsize=(5 * (T_b + 2), B * 5))
        kwargs = dict(cmap=cmap, norm=norm, interpolation="none")
        std_kwargs = dict(cmap=cmap_std, norm=norm_std, interpolation="none")
        for i in range(B):
            t_ctx = self.t_ctx.reshape(B, T_b - 1, -1, 1)
            for j in range(T_b - 1):
                axs[i, j].imshow(f_ctx[i, j], **kwargs)
                axs[i, j].set_xlabel(f"t={t_ctx[i, j, 0, 0].item()}", fontsize=30)
                if i == 0:
                    axs[i, j].set_title("Context", fontsize=30)
            t_test = self.t_test[i, 0, 0].item()
            axs[i, j + 1].imshow(f_test[i, 0], **kwargs)
            axs[i, j + 1].set_xlabel(f"t={t_test}", fontsize=30)
            axs[i, j + 2].imshow(f_pred[i, 0], **kwargs)
            axs[i, j + 2].set_xlabel(f"t={t_test}", fontsize=30)
            axs[i, j + 3].imshow(f_std[i, 0], **std_kwargs)
            axs[i, j + 3].set_xlabel(f"t={t_test}", fontsize=30)
            if i == 0:
                axs[i, j + 1].set_title("Ground Truth", fontsize=30)
                axs[i, j + 2].set_title("Prediction", fontsize=30)
                axs[i, j + 3].set_title("Uncertainty", fontsize=30)
            for j in range(T_b + 2):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
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
            d.mask_ctx,
            d.x_test,
            d.s_test,
            d.t_test,
            d.f_test,
            d.mask_test,
            d.inv_permute_idx,
        ),
        (d.s_dims, d.forecast),
    ),
    lambda aux, children: SpatiotemporalBatch(*children, *aux),
)
