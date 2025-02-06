#!/usr/bin/env python3
import math
from functools import partial
from pathlib import Path
from time import time

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.meta_learning.train_utils import (
    Callback,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    load_ckpt,
    log_img_plots,
    save_ckpt,
    select_steps,
    train,
)
from dl4bi.meta_learning.transform import pointwise_multinomial

# Example command to evaluate only:
# python sir.py \
#   model=icml/tnp_kr_scan \
#   data=128x128 \
#   data.batch_size=1 \
#   valid_num_steps=100 \
#   wandb=False \
#   evaluate_only=True


@hydra.main("configs/sir", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_dataloader(cfg.data, cfg.sim)
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
    train_step, valid_step = select_steps(model, is_categorical=True)
    if cfg.evaluate_only:
        state, _ = load_ckpt(path.with_suffix(".ckpt"))
        # run once to compile
        evaluate(rng_test, state, valid_step, dataloader, num_steps=1)
        start = time()
        metrics = evaluate(
            rng_test,
            state,
            valid_step,
            dataloader,
            cfg.valid_num_steps,
        )
        end = time()
        metrics["time_elapsed_s"] = end - start
        return print(metrics)
    dims = [dim.num for dim in cfg.data.s]
    clbk = partial(
        log_img_plots,
        shape=(*dims, 3),
        num_plots=cfg.data.batch_size,
        remap_colors=remap_colors,
        transform_model_output=pointwise_multinomial,
    )
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        valid_step,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(clbk, cfg.plot_interval)],
    )
    metrics = evaluate(
        rng_test,
        state,
        valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig, priors: DictConfig):
    """A 2D Lattice SIR dataloader."""
    sir = instantiate(priors)
    dims, D_s = tuple([dim.num for dim in data.s]), len(data.s)
    Lc_min, Lc_max, Lt = data.num_ctx.min, data.num_ctx.max, data.num_test
    L, B, N = math.prod(dims), data.batch_size, data.num_steps
    s_grid = build_grid(data.s).reshape(-1, D_s)  # flatten spatial dims
    s = jnp.repeat(s_grid[None, ...], B, axis=0)
    valid_lens_test = jnp.repeat(Lt, B)

    @jit
    def transform_and_permute(rng: jax.Array, steps: jax.Array):
        # convert RSI categories to GBR (RGB) one-hot color vectors;
        # serves dual purpose of one-hot encoding and easier plotting
        steps = rsi_to_rgb(steps)
        permute_steps_idx = random.choice(rng, N, (N,), replace=False)
        steps = steps[permute_steps_idx]
        return steps

    @jit
    def create_batch(rng: jax.Array, steps: jax.Array):
        rng_permute, rng_valid = random.split(rng)
        permute_idx = random.choice(rng_permute, L, (L,), replace=False)
        inv_permute_idx = jnp.argsort(permute_idx)
        valid_lens_ctx = random.randint(rng_valid, (B,), Lc_min, Lc_max)
        s_i = s[:, permute_idx, :]
        f_i = steps[:, permute_idx, :]
        return s_i[:, :Lt, :], f_i[:, :Lt, :], valid_lens_ctx, inv_permute_idx

    def dataloader(rng: jax.Array):
        while True:
            steps = None  # signal garbage collect
            rng_sim, rng_tx_pre, rng = random.split(rng, 3)
            steps, *_ = sir.simulate(rng_sim, dims, N)
            steps = transform_and_permute(rng_tx_pre, steps)
            for i in range(N // B):
                rng_i, rng = random.split(rng)
                steps_i = steps[i * B : (i + 1) * B].reshape(B, L, 3)
                s_i, f_i, valid_lens_ctx, inv_permute_idx = create_batch(rng_i, steps_i)
                yield (
                    s_i[:, :Lc_max, :],
                    f_i[:, :Lc_max, :],
                    valid_lens_ctx,
                    s_i,
                    f_i,
                    valid_lens_test,
                    s,  # add full originals for use in callbacks, e.g. log_plots
                    steps_i,
                    inv_permute_idx,
                )

    return dataloader


@jit
def rsi_to_rgb(steps: jax.Array):
    steps += 1  # [-1, 0, 1] -> [0, 1, 2]
    # 0 (recovered) => 1 (green)
    # 1 (susceptible) => 2 (blue)
    # 2 (infected) => 0 (red)
    mapping = jnp.array([1, 2, 0])
    rgb_cat = mapping[jnp.int32(steps)]
    # convert RGB categories to one-hot vectors
    return jax.nn.one_hot(rgb_cat, 3)


@jit
def remap_colors(x: jax.Array):
    # palette from https://davidmathlogic.com/colorblind
    C = jnp.array([[216, 27, 96], [0, 77, 64], [30, 136, 229]]) / 255.0
    C = C[None, None, ...]
    return (C * x[..., None]).sum(axis=-2)


if __name__ == "__main__":
    main()
