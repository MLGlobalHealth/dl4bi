#!/usr/bin/env python3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import wandb
from hydra.utils import instantiate
from jax import jit, random
from matplotlib import patches
from matplotlib.colors import Normalize
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    TrainState,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialBatch, SpatialData
from dl4bi.meta_learning.data.utils import inv_permute_L_in_BLD, unbatch_BLD
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/multiscale_2d_gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    path = f"results/{cfg.project}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = valid_dataloader = build_dataloader(cfg.data, cfg.kernel)
    clbk_dataloader = build_dataloader(cfg.data, cfg.kernel, is_callback=True)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    clbk_dataloader = build_multires_2d_grid_dataloader(cfg.data, cfg.kernel)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
        callbacks=[Callback(wandb_multires_2d_plots, cfg.plot_interval)],
        callback_dataloader=clbk_dataloader,
        return_state="last",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig, kernel: DictConfig, is_callback: bool = False):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    B, L, D_s = data.batch_size, data.num_ctx.max + data.num_test, len(data.s1)
    get_min = lambda axes: jnp.array([axis["start"] for axis in axes])
    get_max = lambda axes: jnp.array([axis["stop"] for axis in axes])
    to_extra = lambda d: {k: v.item() for k, v in d.items() if v is not None}
    s1_min, s1_max = get_min(data.s1), get_max(data.s1)
    s2_min, s2_max = get_min(data.s2), get_max(data.s2)
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    def dataloader(rng: jax.Array):
        while True:
            rng_s1, rng_s2, rng_gp, rng_b, rng = random.split(rng, 5)
            s1 = random.uniform(rng_s1, (L // 2, D_s), jnp.float32, s1_min, s1_max)
            s2 = random.uniform(rng_s2, (L // 2, D_s), jnp.float32, s2_min, s2_max)
            s = jnp.concat([s1, s2])
            f, var, ls, period, *_ = gp.simulate(rng_gp, s, B)
            s = batchify(s)
            d = SpatialData(x=None, s=s, f=f)
            yield d.batch(
                rng_b,
                data.num_ctx.min,
                data.num_ctx.max,
                num_test=data.num_test,
                test_includes_ctx=False,
                obs_noise=data.obs_noise,
            )

    return dataloader


def build_multires_2d_grid_dataloader(data: DictConfig, kernel: DictConfig):
    """A custom 2D GP dataloader in which generated context and test points
        reside only on the 2d grid.

    .. note::
        The dataloader used for training and testing uses context points
        on a continuous domain, while this only uses points on a grid for
        visualization purposes.
    """
    B, D_s = data.batch_size, len(data.s1)
    gp = instantiate(kernel)
    # create a new axis with the span of s2 and resolution of s1
    ppus = [points_per_unit(axis) for axis in data.s1]
    s_hires_axes = [update_density(axis, ppu) for axis, ppu in zip(data.s2, ppus)]
    s_hires_g, s1_g, s2_g = map(build_grid, (s_hires_axes, data.s1, data.s2))
    s_hires_flat, s1_flat, s2_flat = map(
        lambda x: x.reshape(-1, D_s), (s_hires_g, s1_g, s2_g)
    )
    s_flat = jnp.concat([s_hires_flat, s1_flat, s2_flat])
    s1_s2_batch = jnp.repeat(jnp.concat([s1_flat, s2_flat])[None, ...], B, axis=0)
    s_hires_batch = jnp.repeat(s_hires_flat[None, ...], B, axis=0)

    # NOTE: uses multiresolution for the task but passes on the full high
    # resolution version in the extra data for plotting
    def dataloader(rng: jax.Array):
        L_hires = s_hires_flat.shape[0]
        while True:
            rng_gp, rng_b, rng = random.split(rng, 3)
            f, var, ls, period, *_ = gp.simulate(rng_gp, s_flat, B)
            f_hires_batch, f_s1_s2_batch = f[:, :L_hires], f[:, L_hires:]
            d = SpatialData(x=None, s=s1_s2_batch, f=f_s1_s2_batch)
            yield (
                d.batch(
                    rng_b,
                    data.num_ctx.min,
                    data.num_ctx.max,
                    num_test=s1_s2_batch.shape[1],  # all points
                    test_includes_ctx=True,
                    obs_noise=data.obs_noise,
                ),
                {
                    "var": var.item(),
                    "ls": ls.item(),
                    "s_hires": s_hires_batch,  # [B, L_hires, 2]
                    "f_hires": f_hires_batch,  # [B, L_hires, 1]
                    "L_s1": s1_flat.shape[0],
                    "s1_axes": data.s1,
                    "s2_axes": data.s2,
                    "s_hires_axes": s_hires_axes,
                },
            )

    return dataloader


def points_per_unit(axis):
    return axis["num"] / (axis["stop"] - axis["start"])


def update_density(axis, ppu):
    num = (axis["stop"] - axis["start"]) * ppu
    return {**axis, "num": int(num)}


def wandb_multires_2d_plots(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatialBatch,
    extra: dict,
    num_plots: int = 8,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    B = batch.s_ctx.shape[0]
    N = min(num_plots or B, B)
    L_s1 = extra.pop("L_s1")
    s1_axes = extra.pop("s1_axes")
    s2_axes = extra.pop("s2_axes")
    s_hires_axes = extra.pop("s_hires_axes")
    s_hires = extra.pop("s_hires")
    f_hires = extra.pop("f_hires")
    rng_dropout, rng_extra = random.split(rng_step)
    # NOTE: use original task, but now predict output in high resolution
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **replace(batch, s_test=s_hires, f_test=f_hires, mask_test=None),
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    f_hires_pred, f_hires_std = output.mu, output.std
    inv_p = batch.inv_permute_idx
    f_ctx = jnp.where(batch.mask_ctx[..., None], batch.f_ctx, jnp.nan)
    f_test = batch.f_test
    f_ctx, f_test = unbatch_BLD([f_ctx, f_test], inv_p.shape[0])
    f_ctx, f_test = inv_permute_L_in_BLD([f_ctx, f_test], inv_p)
    f_ctx_s1, f_test_s1 = f_ctx[:, :L_s1], f_test[:, :L_s1]
    f_ctx_s2, f_test_s2 = f_ctx[:, L_s1:], f_test[:, L_s1:]
    s1_shape = [a["num"] for a in s1_axes]
    s2_shape = [a["num"] for a in s2_axes]
    s_hires_shape = [a["num"] for a in s_hires_axes]
    f_ctx_s1 = f_ctx_s1.reshape(B, *s1_shape)
    f_ctx_s2 = f_ctx_s2.reshape(B, *s2_shape)
    f_test_s1 = f_test_s1.reshape(B, *s1_shape)
    f_test_s2 = f_test_s2.reshape(B, *s2_shape)
    f_hires = f_hires.reshape(B, *s_hires_shape)
    f_hires_pred = f_hires_pred.reshape(B, *s_hires_shape)
    f_hires_std = f_hires_std.reshape(B, *s_hires_shape)
    extent_s1 = to_extent(s1_axes)
    extent_s2 = to_extent(s2_axes)
    _, axs = plt.subplots(N, 5, figsize=(25, N * 5))
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    cmap_s1 = mpl.colormaps.get_cmap("Spectral_r")
    cmap_s1.set_bad((0, 0, 0, 0))  # transparent
    for i in range(N):
        f_min = min(f_hires_pred[i].min(), f_hires[i].min())
        f_max = max(f_hires_pred[i].max(), f_hires[i].max())
        norm = Normalize(f_min, f_max)
        kwargs = dict(cmap=cmap, norm=norm)
        kwargs_s1 = dict(cmap=cmap_s1, norm=norm)
        if i == 0:
            axs[i, 0].set_title("Task Truth", fontsize=30)
            axs[i, 1].set_title("Task", fontsize=30)
            axs[i, 2].set_title("Uncertainty", fontsize=30)
            axs[i, 3].set_title("Prediction", fontsize=30)
            axs[i, 4].set_title("Ground Truth", fontsize=30)
        axs[i, 0].set_ylabel(f"Sample {i + 1}", fontsize=30)
        for j in range(5):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            plot_extent_border(axs[i, j], extent_s1)
        axs[i, 0].imshow(f_test_s2[i], extent=extent_s2, **kwargs)
        axs[i, 0].imshow(f_test_s1[i], extent=extent_s1, **kwargs_s1)
        axs[i, 0].set_xlim(*extent_s2[:2])
        axs[i, 0].set_ylim(*extent_s2[2:])
        axs[i, 1].imshow(f_ctx_s2[i], extent=extent_s2, **kwargs)
        axs[i, 1].imshow(f_ctx_s1[i], extent=extent_s1, **kwargs_s1)
        axs[i, 1].set_xlim(*extent_s2[:2])
        axs[i, 1].set_ylim(*extent_s2[2:])
        axs[i, 2].imshow(f_hires_std[i], extent=extent_s2, cmap="plasma")
        axs[i, 3].imshow(f_hires_pred[i], extent=extent_s2, **kwargs)
        axs[i, 4].imshow(f_hires[i], extent=extent_s2, **kwargs)
    subtitle = ", ".join([f"{k}: {v:.2f}" for k, v in extra.items()])
    plt.suptitle(subtitle + "\n", fontsize=35)
    plt.tight_layout()
    path = f"/tmp/gp_multires_2d_{datetime.now().isoformat()}.png"
    plt.savefig(path)
    wandb.log({f"Step {step}": wandb.Image(path)})


def to_extent(axes):
    return [
        axes[1]["start"],  # left
        axes[1]["stop"],  # right
        axes[0]["start"],  # upper
        axes[0]["stop"],  # lower
    ]


def plot_extent_border(ax, extent):
    left, right, top, bottom = extent
    W = right - left
    H = top - bottom
    rect = patches.Rectangle(
        (left, bottom),
        W,
        H,
        fill=False,
        edgecolor="black",
        linewidth=3,
        linestyle="--",
    )
    ax.add_patch(rect)


if __name__ == "__main__":
    main()
