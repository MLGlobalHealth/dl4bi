#!/usr/bin/env python3
from pathlib import Path

import hydra
from jax import random
from omegaconf import DictConfig

from dsp.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    evaluate,
    load_ckpt,
)


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    rng = random.key(cfg.seed)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    model_state, _ = load_ckpt(Path(path).with_suffix(".ckpt"))
    loss = evaluate(rng, model_state, dataloader, num_steps=5000)
    print(f"Loss: {loss:0.4f}")


if __name__ == "__main__":
    main()
