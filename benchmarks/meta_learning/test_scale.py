#!/usr/bin/env python3
import json
import os
from pathlib import Path
from time import time

import hydra
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig
from tqdm import tqdm

from dl4bi.core.train import load_ckpt
from dl4bi.meta_learning.utils import cfg_to_run_name

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    num_exp = 6
    num_trials = 100
    num_fixed = 100
    rng = random.key(cfg.seed)
    s_dim = len(cfg.data.s)
    min_s = jnp.array([axis["start"] for axis in cfg.data.s])
    max_s = jnp.array([axis["stop"] for axis in cfg.data.s])
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    path = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    state, _ = load_ckpt(path.with_suffix(".ckpt"))
    s = random.uniform(rng, (10**num_exp, s_dim), minval=min_s, maxval=max_s)[None, ...]
    f = random.normal(rng, (10**num_exp, 1))[None, ...]

    @jit
    def apply(s_ctx, f_ctx, s_test):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx=None,
            valid_lens_test=None,
            rngs={"extra": rng},
        )

    results = {"ctx": {}, "test": {}}
    # == vary number of test points
    s_ctx = s[:, :num_fixed, :]
    f_ctx = s[:, :num_fixed, :]
    for exp in tqdm(range(num_exp + 1), desc="test"):  # 10^0 -> 10^num_exp
        num_points = 10**exp
        s_test = s[:, :num_points, :]
        # run once to JIT and if it OOMs, skip to varying context points
        try:
            apply(s_ctx, f_ctx, s_test)
        except Exception:  # OOM
            break
        start = time()
        for i in tqdm(range(num_trials), desc="trial", leave=False):
            apply(s_ctx, f_ctx, s_test)
        stop = time()
        results["test"][num_points] = (stop - start) / num_trials
    # == vary number of context points
    s_test = s[:, :num_fixed, :]
    for exp in tqdm(range(num_exp + 1), desc="ctx"):  # 10^0 -> 10^num_exp
        num_points = 10**exp
        s_ctx = s[:, :num_points, :]
        f_ctx = s[:, :num_points, :]
        # run once to JIT and if it OOMs, skip this and remaining tests
        try:
            apply(s_ctx, f_ctx, s_test)
        except Exception:  # OOM
            with open(path.with_stem(path.stem + "_runtimes.json"), "w") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            return results
        start = time()
        for i in range(num_trials):
            apply(s_ctx, f_ctx, s_test)
        stop = time()
        results["ctx"][num_points] = (stop - start) / num_trials
    with open(path.with_stem(path.stem + "_runtimes.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    return results


if __name__ == "__main__":
    results = main()
    print(results)
