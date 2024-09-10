#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax.numpy as jnp
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf

from dsp.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
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
    rng_data, rng_extra = random.split(rng)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    batch = next(dataloader(rng_data))
    s_ctx, f_ctx, valid_lens_ctx, s_test, *_ = batch
    _, vars = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        training=False,
        mutable="intermediates",
        rngs={"extra": rng_extra},
    )
    jnp.save("intermediates.npy", vars["intermediates"])
    labels = (
        "s_ctx",
        "f_ctx",
        "valid_lens_ctx",
        "s_test",
        "f_test",
        "valid_lens_test",
        "var",
        "ls",
        "period",
    )
    jnp.save("batch.npy", dict(zip(labels, batch)))
    out_path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"


if __name__ == "__main__":
    main()
