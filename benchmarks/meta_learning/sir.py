#!/usr/bin/env python3
import math
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialData
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
from dl4bi.meta_learning.utils import (
    cfg_to_run_name,
    save_batches_for_tabpfn,
    wandb_2d_img_callback,
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    build = build_spatial_dataloader
    if cfg.data.type == "spatiotemporal":
        build = build_spatiotemporal_dataloader
    dataloader, clbk_dataloader = build(cfg.data, cfg.sim)
    train_dataloader = valid_dataloader = dataloader
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
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
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
        callbacks=[Callback(clbk, cfg.plot_interval)],
        callback_dataloader=clbk_dataloader,
        return_state="last",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))
    eval_path = path.parent / f"eval_data.npy"
    save_batches_for_tabpfn(rng_test, valid_dataloader, cfg.valid_num_steps, eval_path)


def build_spatial_dataloader(data: DictConfig, priors: DictConfig):
    """A 2D Lattice SIR dataloader over space only."""
    B = data.batch_size
    sir = instantiate(priors)
    s = build_grid(data.s)
    s = jnp.repeat(s[None, ...], data.num_steps, axis=0)
    dims = tuple(axis.num for axis in data.s)
    L = math.prod(dims)

    def dataloader(rng: jax.Array, is_callback: bool = False):
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
                    L if is_callback else data.num_test,
                    test_includes_ctx=True,
                    batch_size=B,
                )

    return dataloader, partial(dataloader, is_callback=True)


def build_spatiotemporal_dataloader(data: DictConfig, priors: DictConfig):
    """A 2D Lattice SIR dataloader over space and time."""
    B = data.batch_size
    sir = instantiate(priors)
    s = build_grid(data.s)
    s = jnp.repeat(s[None, ...], data.num_steps, axis=0)
    t = jnp.arange(data.num_steps)
    dims = tuple(axis.num for axis in data.s)
    L = math.prod(dims)

    def dataloader(rng: jax.Array, is_callback: bool = False):
        while True:
            steps = None  # signal garbage collection
            rng_i, rng = random.split(rng)
            steps, *_ = sir.simulate(rng_i, dims, data.num_steps)
            steps = rsi_to_rgb(steps)
            d = SpatiotemporalData(x=None, s=s, t=t, f=steps)
            for b in range(data.num_steps // B):
                rng_b, rng = random.split(rng)
                yield d.batch(
                    rng_b,
                    data.num_t,
                    data.random_t,
                    data.num_ctx_per_t.min,
                    data.num_ctx_per_t.max,
                    data.independent_t_masks,
                    L if is_callback else data.num_test,
                    data.forecast,
                    data.batch_size,
                )

    return dataloader, partial(dataloader, is_callback=True)


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
    C = C[None, None, None, ...]
    return (C * x[..., None]).sum(axis=-2)


if __name__ == "__main__":
    main()
