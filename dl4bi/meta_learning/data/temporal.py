from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, random
from jax.scipy.stats import norm

from .utils import (
    MetaLearningBatch,
    MetaLearningData,
    batch_BLD,
    permute_L_in_BLD,
)


@dataclass(frozen=True, eq=False)
class TemporalData(MetaLearningData):
    x: Optional[jax.Array]  # [B, T, D_x] or [B, D_x] or None
    t: jax.Array  # [B, T]
    f: jax.Array  # [B, T, D_f]

    def batch(
        self,
        rng: jax.Array,
        num_ctx_min: int,
        num_ctx_max: int,
        num_test: int,
        test_includes_ctx: bool = False,
        obs_noise: Optional[float] = None,
        batch_size: Optional[int] = None,
    ):
        return _batch(
            rng,
            self.x,
            self.t,
            self.f,
            num_ctx_min,
            num_ctx_max,
            num_test,
            test_includes_ctx,
            obs_noise,
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
        "batch_size",
    ),
)
def _batch(
    rng,
    x: Optional[jax.Array],
    t: jax.Array,
    f: jax.Array,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
    test_includes_ctx: bool,
    obs_noise: Optional[float] = None,
    batch_size: Optional[int] = None,
):
    rng_i, rng_p, rng_b, rng_eps = random.split(rng, 4)
    has_x = x is not None
    full_x = has_x and x.shape[:-1] == t.shape
    broadcast_x = has_x and x.ndim == 2
    batch_args = (num_ctx_min, num_ctx_max, num_test, test_includes_ctx)
    t = t[..., None]
    if batch_size is not None:
        idx = random.choice(rng_i, f.shape[0], (batch_size,), replace=False)
        x, t, f = x[idx] if has_x else None, t[idx], f[idx]
    if full_x:
        x, t, f, inv_permute_idx = permute_L_in_BLD(rng_p, [x, t, f])
        args = batch_BLD(rng_b, [x, t, f], *batch_args)
    elif broadcast_x:
        t, f, inv_permute_idx = permute_L_in_BLD(rng_p, [t, f])
        args = batch_BLD(rng_b, [t, f], *batch_args)
        t_ctx, f_ctx, m_ctx, *rest = args
        x_ctx = jnp.repeat(x[:, None], num_ctx_max, axis=1)
        x_test = jnp.repeat(x[:, None], num_test, axis=1)
        args = (x_ctx, t_ctx, f_ctx, m_ctx, x_test, *rest)
    else:
        t, f, inv_permute_idx = permute_L_in_BLD(rng_p, [t, f])
        args = batch_BLD(rng_b, [t, f], *batch_args)
        t_ctx, f_ctx, m_ctx, *rest = args
        args = (None, t_ctx, f_ctx, m_ctx, None, *rest)
    if obs_noise:
        x_ctx, s_ctx, f_ctx, *rest = args
        f_ctx += obs_noise * random.normal(rng_eps, f_ctx.shape)
        args = (x_ctx, s_ctx, f_ctx, *rest)
    return TemporalBatch(*args, inv_permute_idx=inv_permute_idx)


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TemporalData,
    lambda d: ((d.x, d.t, d.f), None),
    lambda _aux, children: TemporalData(*children),
)


@dataclass(frozen=True, eq=False)
class TemporalBatch(MetaLearningBatch):
    x_ctx: Optional[jax.Array]  # [B, L_ctx, D_x] or None
    t_ctx: jax.Array  # [B, L_ctx, 1]
    f_ctx: jax.Array  # [B, L_ctx, D_f]
    mask_ctx: jax.Array  # [B, L_ctx]
    x_test: Optional[jax.Array]  # [B, L_test, D_x]
    t_test: jax.Array  # [B, L_test, 1]
    f_test: jax.Array  # [B, L_test, D_f]
    mask_test: Optional[jax.Array]  # [B, L_test]
    inv_permute_idx: jax.Array  # [L]

    def plot_1d(
        self,
        f_pred: jax.Array,  # [B, [K]?, L_test, 1]
        f_std: jax.Array,  # [B, [K]?, L_test, 1]
        hdi_prob: float = 0.95,
        subtitle: Optional[str] = None,
        num_plots: Optional[int] = None,
        **kwargs,
    ):
        B = self.f_test.shape[0]
        N = min(num_plots or B, B)
        order = jnp.argsort(self.t_test, axis=1)
        arrays = [self.t_test, self.f_test, f_pred, f_std]
        arrays = [jnp.take_along_axis(a, order, axis=1) for a in arrays]
        t_test, f_test, f_pred, f_std = map(lambda v: v[..., 0], arrays)
        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        f_lower, f_upper = f_pred - z_score * f_std, f_pred + z_score * f_std
        _, axs = plt.subplots(N, 1, figsize=(8, N * 4))
        for i in range(N):
            if i == 0:
                title = "Temporal Posterior Predictive"
                title += f"\n{subtitle}" if subtitle else ""
                axs[i].set_title(title, fontsize=16)
            elif i == B - 1:
                axs[i].set_xlabel("t", fontsize=14)
            axs[i].set_ylabel(f"Sample {i + 1}", fontsize=14, rotation=90)
            axs[i].scatter(
                self.t_ctx[i, self.mask_ctx[i], 0],
                self.f_ctx[i, self.mask_ctx[i], 0],
                color="black",
            )
            axs[i].plot(t_test[i], f_test[i], color="black")
            axs[i].plot(t_test[i], f_pred[i], color="steelblue")
            axs[i].fill_between(
                t_test[i],
                f_lower[i],
                f_upper[i],
                alpha=0.4,
                color="steelblue",
                interpolate=True,
            )
        plt.tight_layout()
        return plt.gcf()


# register to use in jitted functions
jax.tree_util.register_pytree_node(
    TemporalBatch,
    lambda d: (
        (
            d.x_ctx,
            d.t_ctx,
            d.f_ctx,
            d.mask_ctx,
            d.x_test,
            d.t_test,
            d.f_test,
            d.mask_test,
            d.inv_permute_idx,
        ),
        None,
    ),
    lambda _aux, children: TemporalBatch(*children),
)
