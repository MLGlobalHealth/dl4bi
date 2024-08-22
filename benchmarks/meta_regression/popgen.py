#!/usr/bin/env python3
import os
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dsp.meta_regression.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_img_plots,
    save_ckpt,
    train,
)

# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#code-generation-flags
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@hydra.main("configs/popgen", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = Path(f"results/popgen/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
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
    model = instantiate(cfg.model)  # TODO(danj): adapt for continue training
    img_cbk = Callback(partial(log_img_plots, shape=(32, 32, 1)), cfg.plot_interval)
    save_cbk = Callback(
        lambda step, rng_step, state, *_: save_ckpt(
            state, cfg, path.with_suffix(".ckpt")
        ),
        cfg.valid_interval,
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
        callbacks=[img_cbk, save_cbk],
    )
    loss = evaluate(rng_test, state, valid_dataloader, cfg.valid_num_steps)
    wandb.log({"test_loss": loss})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(
    batch_size: int = 16,
    num_ctx_min: int = 64,
    num_ctx_max: int = 512,
    num_test_max: int = 1024,
):
    B, L = batch_size, 32 * 32
    # load & convert from [0, 1] -> [-1, 1]
    path = "cache/popgen/f_test_n1000_mu_1e-5_m_5e-3.npy"
    train_ds = 2 * (np.load(path, mmap_mode="r") - 0.5)
    s_test = build_grid([dict(start=-2.0, stop=2.0, num=32)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(num_test_max, B)

    def build_dataloader(dataset):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
                batch_idx = random.choice(rng_batch, N, (B,), replace=False)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                f_test = dataset[batch_idx]
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

    return build_dataloader(train_ds)


def build_scheduled_dataloader(batch_size: int, valid_lens_ctx_schedule: jax.Array):
    B, L = batch_size, 32 * 32
    train_ds = np.load("cache/popgen/f_test_n1000_mu_1e-5_m_5e-3.npy", mmap_mode="r")
    s_test = build_grid([dict(start=-2.0, stop=2.0, num=32)] * 2).reshape(L, 2)
    s_test = jnp.repeat(s_test[None, ...], B, axis=0)  # [L, 2] -> [B, L, 2]
    valid_lens_test = jnp.repeat(L, B)

    def build_dataloader(dataset):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            i = 0
            while True:
                rng_batch, rng_permute, rng_valid, rng = random.split(rng, 4)
                batch_idx = random.choice(rng_batch, N, (B,), replace=False)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                f_test = dataset[batch_idx]
                f_test = f_test.reshape(B, -1, 1)  # [B, H, W, 1] -> [B, L, 1]
                inv_permute_idx = jnp.argsort(permute_idx)
                # permute the order and select the first valid_lens_ctx for context
                s_test_permuted = s_test[:, permute_idx, :]
                f_test_permuted = f_test[:, permute_idx, :]
                valid_lens_ctx = jnp.repeat(valid_lens_ctx_schedule[i], B)
                i += 1
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

    return build_dataloader(train_ds)


def build_valid_lens_ctx_schedule(
    rng: jax.Array,
    num_steps: int = 100000,
    pct_random: float = 0.25,
    num_cycles: int = 3,
    num_ctx_min: int = 64,
    num_ctx_max: int = 1024,
):
    num_sine = int((1.0 - pct_random) * num_steps)
    num_random = num_steps - num_sine
    sine_schedule = build_sine_schedule(num_sine, num_cycles, num_ctx_min, num_ctx_max)
    random_schedule = random.randint(rng, (num_random,), num_ctx_min, num_ctx_max)
    return jnp.hstack([sine_schedule, random_schedule])


def build_sine_schedule(
    num_steps: int,
    num_cycles: int,
    num_ctx_min: int,
    num_ctx_max: int,
):
    scale = (num_ctx_max - num_ctx_min) // 2
    x = jnp.linspace(0, num_cycles * 2 * jnp.pi, num_steps)
    shifted_sine = 1 + jnp.sin(x)
    return num_ctx_min + jnp.astype(scale * shifted_sine, np.int32)


if __name__ == "__main__":
    main()
