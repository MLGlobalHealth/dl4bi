#!/usr/bin/env python3
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Generator

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optax
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid
import json
import networkx as nx

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_graph_plots,
    log_temporal_img_plots,
    save_ckpt,
    train,
)


@hydra.main("configs/outbreaks", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader = build_dataloader(cfg.data_path, cfg.file_name, cfg.graph_dist, cfg.temporal_data, cfg.batch_size)
    # plot_temporal_dataloader(train_dataloader(rng))
    valid_dataloader = build_dataloader(cfg.data_path, cfg.file_name, cfg.graph_dist, cfg.temporal_data, cfg.batch_size)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    # cmap = mpl.colormaps.get_cmap("grey")
    cmap = mpl.colormaps.get_cmap("coolwarm")
    # cmap.set_bad("blue")
    cmap.set_bad("black")
    norm = mpl.colors.Normalize(vmin=cfg.vmin, vmax=cfg.vmax, clip=True)
    with open(cfg.data_path + 'graph_pos.json', 'r') as infile:
            pos = json.load(infile)
    graph = nx.read_adjlist(cfg.data_path + 'graph.adjlist')
    if 'lattice' in cfg.data_path:
        img_cbk = Callback(
        partial(log_temporal_img_plots, shape=(16, 16, 1), cmap=cmap, norm=norm),
        cfg.plot_interval,
    )
    else:
        img_cbk = Callback(
            partial(log_graph_plots, pos=pos, graph=graph, norm=norm),
            cfg.plot_interval,
        )
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[img_cbk],
    )
    metrics = evaluate(rng_test, state, valid_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    data_path: str,
    file_name: str,
    graph_dist_path: str,
    temporal_cfg: DictConfig,
    batch_size: int = 16,
    min_ctx_valid_pct: float = 0.05,
    max_ctx_valid_pct: float = 0.5,
    num_test_max: int = 256,
):
    path = data_path +  file_name # contains [time, f_test]
    dataset = np.load(path, mmap_mode="r")['outbreaks']
    dataset = dataset[:, 1:]  # remove sim_id
    B = batch_size
    L = dataset.shape[1] - 1
    if 'SB_high' in path:
        time_frame_size = 60
    else:
        time_frame_size = 100
    # time_frame_size = dataset.shape[0] // 100000
    assert dataset.shape[0] % time_frame_size == 0
    print('dataset shape:', dataset.shape)
    print('time_frame_size:', time_frame_size)
    print('L:', L)
    
    graph_dist_path = data_path + graph_dist_path
    graph_dist = jnp.load(graph_dist_path)
    
    num_samples = int(dataset.shape[0] // time_frame_size)
    # s_grid = build_grid(
    #     [dict(start=-2.0, stop=2.0, num=16)] * 2 + [dict(start=1, stop=1, num=1)]
    # ).reshape(L, 3)
    s_grid = build_grid([dict(start=-2.0, stop=2.0, num=int(np.ceil(np.sqrt(L))))] * 2 + [dict(start=1, stop=1, num=1)]).reshape(-1,1)[:L* 3].reshape(L, 3)
    
    s_grid = jnp.repeat(s_grid[None, ...], B, axis=0)  # [L, 3] -> [B, L, 3]
    valid_lens_test = jnp.repeat(num_test_max, B)
    dataset = dataset.reshape(num_samples, time_frame_size, -1)
    # dataset = 2 * (dataset - 0.5)  # [0, 1] -> [-1, 1]
    offset_range = jnp.arange(1, temporal_cfg.max_past_time_offset + 1)

    # NOTE this function to prevent repetitions
    def sample_offset(rng_s, ctx_num):
        return random.choice(rng_s, offset_range, shape=(ctx_num,), replace=False)

    def dataloader(rng: jax.Array):
        while True:
            rng_b, rng_perm, rng_val, rng_t1, rng_t2, rng_t3, rng = random.split(rng, 7)
            batch_idx = random.choice(rng_b, num_samples, (B,), replace=False)
            batch = dataset[batch_idx]
            # choosing the number of context time steps
            time_step_ctx_num = random.randint(
                rng_t1,
                shape=(1,),
                minval=temporal_cfg.num_time_ctx[0].min,
                maxval=temporal_cfg.num_time_ctx[0].max + 1,
            )[0].item()
            # choosing the test times for each sample in batch
            test_times = random.randint(
                rng_t2,
                shape=(B, 1),
                minval=temporal_cfg.max_past_time_offset,
                maxval=time_frame_size,
            )
            # choosing specific ctx times for each sample in batch
            ctx_times = (
                -jax.vmap(sample_offset, in_axes=(0, None))(
                    jax.random.split(rng_t3, B), time_step_ctx_num
                )
                + test_times
            )
            s_test = s_grid.at[..., -1].set(s_grid[..., -1] * test_times)
            s_ctx = jnp.repeat(s_grid, time_step_ctx_num, axis=0)
            s_ctx = (
                s_ctx.at[..., -1]
                .set(s_ctx[..., -1] * ctx_times.flatten()[:, None])
                .reshape(B, L * time_step_ctx_num, 3)
            )
            f_test = jnp.take_along_axis(
                batch,
                test_times[..., None],
                axis=1,
            )[..., 0, 1:]
            f_ctx = jnp.take_along_axis(
                batch,
                ctx_times[..., None],
                axis=1,
            )[..., 1:].reshape(B, -1)
            permute_idx_ctx = random.permutation(rng_perm, L * time_step_ctx_num)
            permute_idx_test = random.permutation(rng_perm, L)
            inv_permute_idx_ctx = jnp.argsort(permute_idx_ctx)
            inv_permute_idx_test = jnp.argsort(permute_idx_test)
            valid_lens_ctx = random.randint(
                rng_val,
                (B,),
                int(min_ctx_valid_pct * L * time_step_ctx_num),
                int(max_ctx_valid_pct * L * time_step_ctx_num),
            )
            yield (
                s_ctx[:, permute_idx_ctx, :],  # s_ctx (permuted)
                f_ctx[:, permute_idx_ctx, None],  # f_ctx (permuted)
                valid_lens_ctx,  # only the first valid lens are used/observed
                s_test[:, permute_idx_test, :],  # s_test (permuted)
                f_test[..., permute_idx_test, None],  # f_test (permuted)
                valid_lens_test,
                s_test,  # add full originals for use in callbacks, e.g. log_plots
                f_test[..., None],
                (inv_permute_idx_ctx, inv_permute_idx_test),
                graph_dist
            )

    return dataloader


def plot_temporal_dataloader(dataloader: Generator, sample_idx: int = 0):
    """
    Plots context and test data for a single sample in the batch.
    Verifies that permutation, and time sampling is valid for test and
    context.

    Args:
        dataloader (Generator): The dataloader function.
        sample_idx (int): Index of the sample in the batch to plot.
    """
    (
        s_ctx_permuted,
        f_ctx_permuted,
        valid_lens_ctx,
        _,
        f_test_permuted,
        _,
        s_test,
        f_test,
        (inv_permute_idx_ctx, inv_permute_idx_test),
    ) = next(dataloader)
    num_ctx, img_size = s_ctx_permuted.shape[1] // 256, 16
    ctx_times = s_ctx_permuted[sample_idx, :, -1].reshape(num_ctx, -1)
    test_time = s_test[sample_idx, 0, -1]
    valid_len_ctx = valid_lens_ctx[sample_idx]
    f_ctx_permuted = f_ctx_permuted.at[sample_idx, valid_len_ctx:].set(np.nan)
    f_ctx_unperm_images = f_ctx_permuted[
        sample_idx, inv_permute_idx_ctx
    ].reshape(-1, img_size, img_size)
    f_test_perm_image = f_test_permuted[sample_idx].reshape(img_size, img_size)
    f_test_unperm_image = f_test_permuted[
        sample_idx, inv_permute_idx_test
    ].reshape(img_size, img_size)
    f_test_direct_image = f_test[sample_idx].reshape(img_size, img_size)
    fig, axes = plt.subplots(2, max(3, num_ctx), figsize=(5 * max(3, num_ctx), 12))
    cmap = mpl.colormaps.get_cmap("gray").copy()
    cmap.set_bad(color="blue")

    def plot_single(ax, im, title):
        ax.imshow(im / 2 + 0.5, origin="lower", cmap=cmap)
        ax.set_title(title)

    for i in range(num_ctx):
        plot_single(
            axes[0, i],
            f_ctx_unperm_images[i],
            f"Unperm. Context Timestep {ctx_times[i, 0]:.2f}",
        )
    plot_single(axes[-1, 0], f_test_perm_image, f"Perm Test at time {test_time:.2f}")
    plot_single(
        axes[-1, 1], f_test_unperm_image, f"Unperm. Test at time {test_time:.2f}"
    )
    plot_single(axes[-1, 2], f_test_direct_image, f"Full Test at time {test_time:.2f}")
    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    save_path = f"cache/outbreaks/temporal_plot_{datetime.now().isoformat()}.png"
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
