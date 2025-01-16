#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import optax
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf

from dl4bi.meta_regression.train_utils import (
    Callback,
    build_gp_dataloader,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_2d_grid_gp_plots,
    log_posterior_predictive_plots,
    save_ckpt,
    select_steps,
    train,
)


@hydra.main("configs/gp", config_name="default", version_base=None)
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
    clbk = log_posterior_predictive_plots
    if cfg.data.name == "2d":
        H, W = cfg.data.s[0].num, cfg.data.s[1].num
        clbk = partial(
            log_2d_grid_gp_plots,
            shape=(H, W, 1),
            data=cfg.data,
            kernel=cfg.kernel,
        )
    train_step, valid_step = select_steps(model)
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
    path = f"results/{cfg.project}/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


if __name__ == "__main__":
    main()
