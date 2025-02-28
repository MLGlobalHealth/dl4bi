from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from jax import jit, random
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
class SpatialData(MetaLearningData):
    x: Optional[jax.Array]  # [B, [S]+, D_x] or [B, D_x] or None
    s: jax.Array  # [B, [S]+, D_s]
    f: jax.Array  # [B, [S]+, D_f]

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
            self.s,
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
    rng,
    x: Optional[jax.Array],
    s: jax.Array,
    f: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool,
):
    rng_p, rng_b = random.split(rng)
    has_x = x is not None
    broadcast_x = x.ndim == 2 if has_x else None
    s, f = flatten_spatial(s), flatten_spatial(f)
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    if broadcast_x or not has_x:
        x_ctx = x_test = x
        s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [s, f])
        args = batch_BLD(rng_b, [s, f], *batch_args)
        s_ctx, f_ctx, v_ctx, s_test, f_test, *rest = args
        if broadcast_x:
            x_ctx = jnp.repeat(x_ctx[:, None], num_ctx_max, axis=1)
            x_test = jnp.repeat(x_test[:, None], num_test, axis=1)
        args = (x_ctx, s_ctx, f_ctx, v_ctx, x_test, s_test, f_test, *rest)
    else:
        x = flatten_spatial(x)
        x, s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [x, s, f])
        args = batch_BLD(rng_b, [x, s, f], *batch_args)
    return SpatialBatch(*args, inv_permute_idx=inv_permute_idx, s_shape=s.shape)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    SpatialData,
    lambda d: ((d.x, d.s, d.f), None),
    lambda _aux, children: SpatialData(*children),
)


@dataclass(frozen=True, eq=False)
class SpatialBatch(MetaLearningBatch):
    x_ctx: Optional[jax.Array]  # [B, L_ctx, D_x] or None
    s_ctx: jax.Array  # [B, L_ctx, D_s]
    f_ctx: jax.Array  # [B, L_ctx, D_f]
    valid_lens_ctx: jax.Array  # [B]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    s_test: jax.Array  # [B, L_test, D_s]
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
    SpatialBatch,
    lambda d: (
        (
            d.x_ctx,
            d.s_ctx,
            d.f_ctx,
            d.valid_lens_ctx,
            d.x_test,
            d.s_test,
            d.f_test,
            d.valid_lens_test,
            d.inv_permute_idx,
        ),
        (d.s_shape,),
    ),
    lambda aux, children: SpatialBatch(*children, *aux),
)
