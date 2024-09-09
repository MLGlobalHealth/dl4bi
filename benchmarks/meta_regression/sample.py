#!/usr/bin/env python3
from pathlib import Path

import hydra
import imageio.v3 as iio
import jax.numpy as jnp
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from pygifsicle import optimize

from dsp.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
    sample_gif,
)


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="sample_default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.get("project", "Sampling"),
        reinit=True,  # allows reinitialization for multiple runs
    )
    cfg.data.batch_size = 1  # override GP batch argument
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_data, rng_sample = random.split(rng)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    batch = next(dataloader(rng_data))
    batch = _trim_test(batch, cfg.data.num_ctx.min, cfg.data.num_ctx.max)
    paths = sample_gif(rng, state, batch)
    out_path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    for path_id, samples in paths.items():
        for sample_id, sample_paths in samples.items():
            gif_path = f"{out_path} - Path {path_id}, Sample {sample_id}.gif"
            frames = jnp.stack([iio.imread(p) for p in sample_paths], axis=0)
            iio.imwrite(gif_path, frames)
            optimize(gif_path)


def _trim_test(batch, min_ctx, max_ctx):
    """Only get random points non-context from test set, ignore the linear points."""
    s_ctx, f_ctx, _, s_test, f_test, _, *rest = batch
    s_test, f_test = s_test[:, min_ctx:max_ctx, :], f_test[:, min_ctx:max_ctx, :]
    B = s_ctx.shape[0]
    valid_lens_ctx = jnp.repeat(min_ctx, B)
    valid_lens_test = jnp.repeat(max_ctx - min_ctx, B)
    return (s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *rest)


if __name__ == "__main__":
    main()
