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
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dsp.meta_regression.train_utils import (
    Callback,
    log_img_plots,
    save_ckpt,
    train,
    validate,
    cosine_annealing_lr,
)


@hydra.main("configs/mnist", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    model_cfg_name = d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", model_cfg_name),
        project="SPTx - MNIST",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_valid = random.split(rng)
    train_dataloader, valid_dataloader = build_dataloaders()
    train_num_steps, valid_num_steps = 200000, None  # exhaust valid dataloader
    valid_interval, plot_interval = 25000, 50000
    lr_peak, lr_pct_warmup = 5e-4, 0.3
    lr_schedule = cosine_annealing_lr(train_num_steps, lr_peak, lr_pct_warmup)
    optimizer = optax.yogi(lr_schedule)
    state = train(
        rng_train,
        cfg.model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        train_num_steps,
        valid_num_steps,
        valid_interval,
        callbacks=[Callback(partial(log_img_plots, shape=(28, 28)), plot_interval)],
    )
    path = Path(f"results/mnist/{model_cfg_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    loss = validate(
        rng_valid,
        state,
        valid_dataloader,
        valid_num_steps,
        path.with_suffix(".pkl"),
    )
    wandb.log({"test_loss": loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    batch_size: int = 64,
    num_ctx_min: int = 3,
    num_ctx_max: int = 200,
    num_test_max: int = 400,
):
    B, L = batch_size, 28 * 28
    normalize = lambda sample: tf.cast(sample["image"], tf.float32) / 255.0
    train_ds = tfds.load("mnist", split="train").map(normalize)
    valid_ds = tfds.load("mnist", split="test").map(normalize)
    train_ds = train_ds.repeat().batch(batch_size).prefetch(1)
    valid_ds = valid_ds.batch(batch_size).prefetch(1)
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
