#!/usr/bin/env python3
from pathlib import Path

import hydra
from gp import build_gp_dataloader
from jax import random
from omegaconf import DictConfig

from dl4bi.meta_learning.train_utils import (
    cfg_to_run_name,
    evaluate,
    load_ckpt,
)


# NOTE: use the same configs as the Gaussian Process (GP) models
@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    kernel = cfg.kernel.kwargs.kernel.func
    path = Path(f"results/gp/{cfg.data.name}/{kernel}/{cfg.seed}/{run_name}")
    rng = random.key(cfg.seed)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    model_state, _ = load_ckpt(path.with_suffix(".ckpt"))
    num_steps = 5000
    results_path = path.with_stem(path.stem + "_eval.pkl")
    loss = evaluate(rng, model_state, dataloader, num_steps, results_path)
    print(f"\n\nResults saved to: {results_path}")
    print(f"Loss: {loss:0.4f}")


if __name__ == "__main__":
    main()
