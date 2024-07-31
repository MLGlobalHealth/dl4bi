#!/usr/bin/env python3
from pathlib import Path

import hydra
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf

from dsp.meta_regression.train_utils import (
    Callback,
    build_gp_dataloader,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_posterior_predictive_plots,
    save_ckpt,
    train,
)


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project="SPTx - Gaussian Processes",
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
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
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(log_posterior_predictive_plots, cfg.plot_interval)],
    )
    loss = evaluate(rng_test, state, dataloader, cfg.valid_num_steps)
    wandb.log({"test_loss": loss})
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


if __name__ == "__main__":
    main()
