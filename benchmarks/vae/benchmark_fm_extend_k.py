#!/usr/bin/env python3
"""benchmark_fm_extend_k.py

Extends benchmark_fm results with FM-DeepRV K=5 and K=10 inference,
reusing the saved K=1 checkpoint weights (no retraining needed).

K only affects the number of Euler steps in decode(); the vector field
weights are trained identically regardless of K.

Run from repo root:
    uv run python benchmarks/vae/benchmark_fm_extend_k.py
"""

import sys

sys.path.append("benchmarks/vae")

import pickle
from pathlib import Path

import jax.numpy as jnp
import optax
from jax import random
from numpyro import distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from orbax.checkpoint import PyTreeCheckpointer
from scipy.stats import wasserstein_distance

import arviz as az
from benchmark_fm import (
    GRIDS,
    HMC_CHAINS,
    HMC_SAMPLES,
    HMC_WARMUP,
    build_inference_model,
    build_spatial_grid,
    collect_result,
    gen_spatial_obs_mask,
    gen_y_obs,
    run_hmc,
    aggregate_and_plot,
)
from dl4bi.core.train import TrainState
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField
from dl4bi.vae.train_utils import generate_surrogate_decoder

FM_K_EXTRA = [5, 10]


def load_fm_state(ckpt_path: Path, s) -> TrainState:
    """Restore FM vector field weights into a fresh TrainState."""
    fm_vf = FlowMatchingVectorField(num_blks=2)
    fm_base = FlowMatchingDeepRV(vf=fm_vf, n_steps=1)
    L = s.shape[0]
    dummy_batch = {
        "s": s,
        "z": jnp.ones((1, L)),
        "conditionals": jnp.array([10.0]),
        "f": jnp.ones((1, L)),
    }
    rngs = {"params": random.key(0), "extra": random.key(1)}
    init_vars = fm_base.init(rngs, **dummy_batch)
    init_params = init_vars.pop("params")
    state_template = TrainState.create(
        apply_fn=fm_base.apply,
        params=init_params,
        kwargs=init_vars,
        tx=optax.adam(1e-3),
    )
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(
        ckpt_path.absolute(), item={"state": state_template, "config": {}}
    )
    return ckpt["state"]


def replay_grid_data(seed: int, gt_ls: int, target_grid_n: int):
    """Replay rng splitting from benchmark_fm.main() to reproduce y_obs and obs_mask."""
    rng = random.key(seed)
    for grid_n in GRIDS:
        rng, rng_obs, rng_mask, *_ = random.split(rng, 6)
        if grid_n == target_grid_n:
            s = build_spatial_grid(grid_n)
            y_obs = gen_y_obs(rng_obs, s, gt_ls)
            obs_mask = gen_spatial_obs_mask(rng_mask, (grid_n, grid_n))
            return s, y_obs, obs_mask
    raise ValueError(f"grid_n={target_grid_n} not in GRIDS={GRIDS}")


def main(seed: int = 42, gt_ls: int = 10):
    save_dir = Path(f"results/poc_ls_{gt_ls}/").resolve()
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    cond_names = list(priors.keys())

    for grid_n in GRIDS:
        s, y_obs, obs_mask = replay_grid_data(seed, gt_ls, grid_n)
        L = s.shape[0]
        grid_dir = save_dir / f"grid_{L}"

        with open(grid_dir / "Baseline_GP" / "hmc_samples.pkl", "rb") as f:
            baseline_samples = pickle.load(f)

        infer_model = build_inference_model(s, priors)
        ckpt_path = grid_dir / "FM-DeepRV_1_step" / "model.ckpt"
        loaded_state = load_fm_state(ckpt_path, s)

        for k in FM_K_EXTRA:
            model_name = f"FM-DeepRV ({k} steps)"
            folder = model_name.replace(" ", "_").replace("(", "").replace(")", "")
            model_dir = grid_dir / folder
            model_dir.mkdir(parents=True, exist_ok=True)

            if (model_dir / "single_res.pkl").exists():
                print(f"  [{model_name} @ {grid_n}×{grid_n}] already done, skipping.")
                continue

            print(f"\n=== {model_name} | {grid_n}×{grid_n} | ls={gt_ls} ===")

            fm_k = FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=k)
            surrogate_decoder = generate_surrogate_decoder(loaded_state, fm_k)

            rng_infer = random.key(seed * 10_000 + k * 100 + grid_n)
            samples, mcmc, post, infer_time = run_hmc(
                rng_infer, infer_model, y_obs, obs_mask, model_dir, surrogate_decoder
            )

            cond_samples = {c: samples[c] for c in cond_names if c in samples}
            res = collect_result(
                model_name, None, infer_time, None,
                None, None, None,
                y_obs, post, obs_mask, cond_samples, mcmc, seed, L,
            )
            for c in cond_names:
                if c in cond_samples and c in baseline_samples:
                    res[f"{c} wasserstein"] = wasserstein_distance(
                        baseline_samples[c], cond_samples[c]
                    )
                else:
                    res[f"{c} wasserstein"] = float("nan")

            with open(model_dir / "single_res.pkl", "wb") as f:
                pickle.dump(res, f)

            print(
                f"  done — infer_time={infer_time:.1f}s, "
                f"ls_mean={float(samples['ls'].mean()):.2f}, "
                f"n_eff={float(res.get('n_eff ls', 0)):.0f}"
            )

    aggregate_and_plot(save_dir)


if __name__ == "__main__":
    main(seed=42, gt_ls=10)
    main(seed=57, gt_ls=20)
