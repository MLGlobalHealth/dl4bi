#!/usr/bin/env python3
from functools import partial
from pathlib import Path
from time import time

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    cosine_annealing_lr,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.utils import cfg_to_run_name, wandb_2d_img_callback

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
    if cfg.evaluate_only:
        state, _ = load_ckpt(path.with_suffix(".ckpt"))
        # run once to compile
        evaluate(rng_test, state, model.valid_step, dataloader, num_steps=1)
        start = time()
        metrics = evaluate(
            rng_test,
            state,
            model.valid_step,
            dataloader,
            cfg.valid_num_steps,
        )
        end = time()
        metrics["time_elapsed_s"] = end - start
        return print(metrics)
    clbk = partial(
        wandb_2d_img_callback,
        filename_prefix="sir",
        remap_colors=remap_colors,
        transform_model_output=lambda x: (x.p, x.std),
    )
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        model.valid_step,
        dataloader,
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
        model.valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig, priors: DictConfig):
    """A 2D Lattice SIR dataloader over space only."""
    B = data.batch_size
    sir = instantiate(priors)
    s = build_grid(data.s)
    s = jnp.repeat(s[None, ...], data.num_steps, axis=0)
    dims = tuple(axis.num for axis in data.s)

    def dataloader(rng: jax.Array):
        while True:
            steps = None  # signal garbage collection
            rng_i, rng = random.split(rng)
            steps, *_ = sir.simulate(rng_i, dims, data.num_steps)
            steps = rsi_to_rgb(steps)
            d = SpatialData(x=None, s=s, f=steps)
            for b in range(data.num_steps // B):
                rng_b, rng = random.split(rng)
                yield d.batch(
                    rng_b,
                    data.num_ctx.min,
                    data.num_ctx.max,
                    data.num_test,
                    test_includes_ctx=True,
                    batch_size=B,
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
