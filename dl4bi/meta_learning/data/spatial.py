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
        obs_noise: Optional[float] = None,
        test_includes_noise: bool = False,
        batch_size: Optional[int] = None,
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
            obs_noise,
            test_includes_noise,
            batch_size,
        )


@partial(
    jit,
    static_argnames=(
        "num_ctx_min",
        "num_ctx_max",
        "num_test",
        "test_includes_ctx",
        "obs_noise",
        "test_includes_noise",
        "batch_size",
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
    test_includes_ctx: bool = False,
    obs_noise: Optional[float] = None,
    test_includes_noise: bool = False,
    batch_size: Optional[int] = None,
):
    rng_i, rng_p, rng_b, rng_eps = random.split(rng, 4)
    has_x = x is not None
    full_x = has_x and x.shape[:-1] == s.shape[:-1]
    broadcast_x = has_x and x.ndim == 2
    S_to_L = jit(lambda v: v.reshape(v.shape[0], -1, v.shape[-1]))
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    if batch_size is not None:
        idx = random.choice(rng_i, f.shape[0], (batch_size,), replace=False)
        x, s, f = x[idx] if has_x else None, s[idx], f[idx]
    s_shape = s.shape
    if full_x:
        x, s, f = map(S_to_L, (x, s, f))
        x, s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [x, s, f])
        args = batch_BLD(rng_b, [x, s, f], *batch_args)
    elif broadcast_x:
        s, f = map(S_to_L, (s, f))
        s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [s, f])
        s_ctx, f_ctx, m_ctx, *rest = batch_BLD(rng_b, [s, f], *batch_args)
        x_ctx = jnp.repeat(x[:, None], num_ctx_max, axis=1)
        x_test = jnp.repeat(x[:, None], num_test, axis=1)
        args = (x_ctx, s_ctx, f_ctx, m_ctx, x_test, *rest)
    else:
        s, f = map(S_to_L, (s, f))
        s, f, inv_permute_idx = permute_L_in_BLD(rng_p, [s, f])
        args = batch_BLD(rng_b, [s, f], *batch_args)
        s_ctx, f_ctx, m_ctx, *rest = args
        args = (None, s_ctx, f_ctx, m_ctx, None, *rest)
    if obs_noise:
        if test_includes_noise:
            rng_eps, rng_test_eps = random.split(rng_eps)
        x_ctx, s_ctx, f_ctx, m_ctx, x_test, s_test, f_test, *rest = args
        f_ctx += obs_noise * random.normal(rng_eps, f_ctx.shape)
        if test_includes_noise:
            f_test += obs_noise * random.normal(rng_test_eps, f_test.shape)
        args = (x_ctx, s_ctx, f_ctx, m_ctx, x_test, s_test, f_test, *rest)

    return SpatialBatch(*args, inv_permute_idx=inv_permute_idx, s_shape=s_shape)


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
    mask_ctx: jax.Array  # [B, L_ctx]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    s_test: jax.Array  # [B, L_test, D_s]
    f_test: jax.Array  # [B, L_test, D_f]
    mask_test: Optional[jax.Array]  # [B, L_test] or None
    inv_permute_idx: jax.Array  # [L]
    s_shape: tuple

    def plot_1d(
        self,
        f_pred: jax.Array,  # [B, L_test, 1]
        f_std: jax.Array,  # [B, L_test, 1]
        hdi_prob: float = 0.95,
        subtitle: Optional[str] = None,
        num_plots: Optional[int] = None,
        **kwargs,
    ):
        B = self.f_test.shape[0]
        N = min(num_plots or B, B)
        order = jnp.argsort(self.s_test, axis=1)
        arrays = [self.s_test, self.f_test, f_pred, f_std]
        arrays = [jnp.take_along_axis(a, order, axis=1) for a in arrays]
        s_test, f_test, f_pred, f_std = map(lambda v: v[..., 0], arrays)
        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        f_lower, f_upper = f_pred - z_score * f_std, f_pred + z_score * f_std
        _, axs = plt.subplots(N, 1, figsize=(8, N * 4))
        for i in range(N):
            if i == 0:
                title = "Spatial Posterior Predictive"
                title += f"\n{subtitle}" if subtitle else ""
                axs[i].set_title(title, fontsize=16)
            elif i == N - 1:
                axs[i].set_xlabel("s", fontsize=14)
            axs[i].set_ylabel(f"Sample {i + 1}", fontsize=14, rotation=90)
            axs[i].scatter(
                self.s_ctx[i, self.mask_ctx[i], 0],
                self.f_ctx[i, self.mask_ctx[i], 0],
                color="black",
            )
            axs[i].plot(s_test[i], f_test[i], color="black")
            axs[i].plot(s_test[i], f_pred[i], color="steelblue")
            axs[i].fill_between(
                s_test[i],
                f_lower[i],
                f_upper[i],
                alpha=0.4,
                color="steelblue",
                interpolate=True,
            )
        plt.tight_layout()
        return plt.gcf()

    def plot_2d(
        self,
        f_pred: jax.Array,  # [B, L_test, D_f]
        f_std: jax.Array,  # [B, L_test, D_f]
        cmap=mpl.colormaps.get_cmap("Spectral_r"),
        cmap_std=mpl.colormaps.get_cmap("Spectral_r"),
        norm=None,
        norm_std=None,
        remap_colors: Callable = lambda x: x,
        subtitle: Optional[str] = None,
        num_plots: Optional[int] = None,
        **kwargs,
    ):
        B = self.f_test.shape[0]
        inv_p = self.inv_permute_idx
        L = inv_p.shape[0] if inv_p.ndim == 1 else inv_p.shape[1]
        N = min(num_plots or B, B)
        f_ctx = jnp.where(self.mask_ctx[..., None], self.f_ctx, jnp.nan)
        f_test = self.f_test
        if f_std.shape[-1] > 1:  # e.g. uncertainty per RGB channel
            f_std = f_std.mean(axis=-1, keepdims=True)
        if self.mask_test is not None:
            f_test = jnp.where(self.mask_test[..., None], self.f_test, jnp.nan)
        reshape = jit(lambda v: v.reshape(*self.s_shape[:-1], v.shape[-1]).squeeze())
        arrays = unbatch_BLD([f_ctx, f_test, f_pred, f_std], L)
        arrays = inv_permute_L_in_BLD(arrays, self.inv_permute_idx)
        f_ctx, f_test, f_pred, f_std = map(reshape, arrays)
        f_ctx, f_test, f_pred = map(remap_colors, [f_ctx, f_test, f_pred])
        _, axs = plt.subplots(N, 4, figsize=(20, N * 5))
        kwargs = dict(cmap=cmap, norm=norm, interpolation="none")
        std_kwargs = dict(cmap=cmap_std, norm=norm_std, interpolation="none")
        for i in range(N):
            if i == 0:
                axs[i, 0].set_title("Task", fontsize=30)
                axs[i, 1].set_title("Uncertainty", fontsize=30)
                axs[i, 2].set_title("Prediction", fontsize=30)
                axs[i, 3].set_title("Ground Truth", fontsize=30)
            axs[i, 0].set_ylabel(f"Sample {i + 1}", fontsize=30)
            for j in range(4):
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
            axs[i, 0].imshow(f_ctx[i], **kwargs)
            axs[i, 1].imshow(f_std[i], **std_kwargs)
            axs[i, 2].imshow(f_pred[i], **kwargs)
            axs[i, 3].imshow(f_test[i], **kwargs)
        if subtitle:
            plt.suptitle(subtitle + "\n", fontsize=30)
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
            d.mask_ctx,
            d.x_test,
            d.s_test,
            d.f_test,
            d.mask_test,
            d.inv_permute_idx,
        ),
        (d.s_shape,),
    ),
    lambda aux, children: SpatialBatch(*children, *aux),
)
