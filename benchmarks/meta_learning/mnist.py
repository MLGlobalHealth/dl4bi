#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.data import SpatialData
from dl4bi.meta_learning.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    regression_to_rgb,
    save_ckpt,
    select_steps,
    train,
)


@hydra.main("configs/mnist", config_name="default", version_base=None)
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
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader, valid_dataloader = build_dataloaders()
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
    clbk = partial(
        log_img_plots,
        shape=(28, 28, 1),
        cmap=cmap,
        remap_colors=regression_to_rgb,
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(clbk, cfg.plot_interval)],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 16,
    buffer_size: int = 1024,
    num_ctx_min: int = 16,
    num_ctx_max: int = 128,
    num_test_max: int = 256,
):
    B = batch_size
    normalize = lambda sample: 2 * (
        tf.cast(sample["image"], tf.float32) / 255.0 - 0.5
    )  # [0, 255] -> [-0.5, 0.5] -> [-1, 1]
    train_ds = tfds.load("mnist", split="train").map(normalize).repeat()
    valid_ds = tfds.load("mnist", split="test").map(normalize)
    train_ds = train_ds.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    s = build_grid([dict(start=-2.0, stop=2.0, num=28)] * 2)
    s = jnp.repeat(s[None, ...], B, axis=0)

    def build_dataloader(dataset):
        def dataloader(rng: jax.Array):
            for f in dataset.as_numpy_iterator():
                rng_i, rng = random.split(rng)
                yield SpatialData(x=None, s=s, f=f).to_batch(
                    rng_i,
                    num_ctx_min,
                    num_ctx_max,
                    num_test_max,
                    test_includes_ctx=True,
                    include_inv_permute_idx=True,
                )

        return dataloader

    return build_dataloader(train_ds), build_dataloader(valid_ds)


if __name__ == "__main__":
    main()
