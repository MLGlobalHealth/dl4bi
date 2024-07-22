#!/usr/bin/env python3
from pathlib import Path

import hydra
import optax
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig, OmegaConf

from dsp.meta_regression.train_utils import (
    Callback,
    build_gp_dataloader,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    log_posterior_predictive_plots,
    save_ckpt,
    train,
)


@hydra.main("configs/gp", version_base=None)
def main(cfg: DictConfig):
    d = HydraConfig.get().runtime.choices
    exp, kernel, model_cfg_name = d["exp"], d["kernel"], d["model"]
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if "wandb" in cfg else "disabled",
        name=cfg.get("name", model_cfg_name),
        project="SPTx - Gaussian Processes",
    )
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_gp_dataloader(cfg.exp, cfg.kernel)
    train_num_steps, valid_num_steps = 100000, 5000
    valid_interval, plot_interval = 25000, 50000
    lr_peak, lr_pct_warmup = 1e-3, 0.3
    lr_schedule = cosine_annealing_lr(train_num_steps, lr_peak, lr_pct_warmup)
    optimizer = optax.yogi(lr_schedule)
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        dataloader,
        dataloader,
        train_num_steps,
        valid_num_steps,
        valid_interval,
        callbacks=[Callback(log_posterior_predictive_plots, plot_interval)],
    )
    loss = evaluate(rng_test, state, dataloader, valid_num_steps)
    wandb.log({"test_loss": loss})
    path = Path(f"results/gp/{exp}-{kernel}-{model_cfg_name}-seed-{cfg.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


if __name__ == "__main__":
    main()
