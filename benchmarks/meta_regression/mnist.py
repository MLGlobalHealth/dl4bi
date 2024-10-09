#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.meta_regression.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
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
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(partial(log_img_plots, shape=(28, 28, 1)), cfg.plot_interval)
        ],
    )
    metrics = evaluate(rng_test, state, valid_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/mnist/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 32,
    buffer_size: int = 1024,
    num_ctx_min: int = 10,
    num_ctx_max: int = 100,
    num_test_max: int = 200,
):
    B, L = batch_size, 28 * 28
    normalize = lambda sample: 2 * (
        tf.cast(sample["image"], tf.float32) / 255.0 - 0.5
    )  # [0, 255] -> [-0.5, 0.5] -> [-1, 1]
    train_ds = tfds.load("mnist", split="train").map(normalize).repeat()
    valid_ds = tfds.load("mnist", split="test").map(normalize)
    train_ds = train_ds.shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    s_test = build_grid([dict(start=-1.0, stop=1.0, num=28)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)  # similar to ANP, Appendix D

    def build_dataloader(dataset):
        def dataloader(rng: jax.Array):
            for f_test in dataset.as_numpy_iterator():
                rng_permute, rng_valid, rng = random.split(rng, 3)
                f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
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

    return build_dataloader(train_ds), build_dataloader(valid_ds)


if __name__ == "__main__":
    main()
