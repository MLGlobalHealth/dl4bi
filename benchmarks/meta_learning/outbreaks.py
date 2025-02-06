#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.meta_learning.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
    select_steps,
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
    train_dataloader = build_dataloader(cfg.batch_size)
    valid_dataloader = build_dataloader(cfg.batch_size)
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
    train_step, valid_step = select_steps(model)
    cmap = mpl.colormaps.get_cmap("grey")
    cmap.set_bad("blue")
    norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=True)
    img_cbk = Callback(
        partial(log_img_plots, shape=(16, 16, 1), cmap=cmap, norm=norm),
        cfg.plot_interval,
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[img_cbk],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    batch_size: int = 16,
    num_ctx_min: int = 16,
    num_ctx_max: int = 128,
    num_test_max: int = 256,
):
    B, L = batch_size, 16 * 16
    path = "cache/outbreaks/outbreaks.npy"  # contains [time, f_test]
    dataset = np.load(path, mmap_mode="r")
    s_grid = build_grid([dict(start=-2.0, stop=2.0, num=16)] * 2).reshape(L, 2)
    s_grid = jnp.repeat(s_grid[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)
    N = dataset.shape[0]

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
            batch_idx = random.choice(rng_batch, N, (B,), replace=False)
            permute_idx = random.choice(rng_permute, L, (L,), replace=False)
            batch = dataset[batch_idx]
            time, f_test = batch[:, [0]], batch[:, 1:]
            time = jnp.repeat(time[:, None, :], L, axis=1)
            f_test = 2 * (f_test - 0.5)  # [0, 1] -> [-1, 1]
            s_test = jnp.concatenate([s_grid, time], axis=-1)
            f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
            inv_permute_idx = jnp.argsort(permute_idx)
            # permute the order and select the first valid_lens_ctx for context
            s_test_permuted = s_test[:, permute_idx, :]
            f_test_permuted = f_test[:, permute_idx, :]
            s_test_permuted = s_test_permuted[:, :num_test_max, :]
            f_test_permuted = f_test_permuted[:, :num_test_max, :]
            valid_lens_ctx = random.randint(
                rng_valid,
                (B,),
                num_ctx_min,
                num_ctx_max,
            )
            yield (
                s_test_permuted,  # s_ctx (permuted)
                f_test_permuted,  # f_ctx (permuted)
                valid_lens_ctx,  # only the first valid lens are used/observed
                s_test_permuted,  # s_test (permuted)
                f_test_permuted,  # f_test (permuted)
                valid_lens_test,
                s_test,  # add full originals for use in callbacks, e.g. log_plots
                f_test,
                inv_permute_idx,
            )

    return dataloader


if __name__ == "__main__":
    main()
