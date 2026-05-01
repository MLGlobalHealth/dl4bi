#!/usr/bin/env python3
"""benchmark_poc.py

Proof-of-concept benchmark: Baseline GP vs DeepRV vs FM-DeepRV (1 and 3 steps)
on a Matérn-1/2 GP prior with Poisson likelihood.

Models:
  - Baseline GP     (exact Cholesky, HMC only)
  - DeepRV + gMLP   (adamw, single forward pass)
  - FM-DeepRV 1-step
  - FM-DeepRV 3-step

Grid sizes : 16×16, 32×32
Lengthscales: 10, 20

Self-contained — generates data on the fly, no precomputed pickles required.
Interruption-safe — skips any model/grid/ls combo that already has single_res.pkl.

Run from the repo root:
    python benchmarks/vae/benchmark_poc.py
"""

import sys

sys.path.append("benchmarks/vae")

import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from scipy.stats import wasserstein_distance
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState,
    cosine_annealing_lr,
    estimate_flops,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField, gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    flow_matching_train_step,
    flow_matching_valid_step,
    generate_surrogate_decoder,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GRIDS = [16, 32]
LENGTHSCALES = [10, 20]
TRAIN_STEPS = 200_000
VALID_INTERVAL = 25_000
VALID_STEPS = 5_000
BATCH_SIZE = 32
MAX_LR = 1e-3
HMC_WARMUP = 4_000
HMC_SAMPLES = 6_000
HMC_CHAINS = 2


# ---------------------------------------------------------------------------
# Valid step for gMLPDeepRV
# ---------------------------------------------------------------------------

@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


# ---------------------------------------------------------------------------
# Data generation (self-contained, no precomputed pickles)
# ---------------------------------------------------------------------------

def build_spatial_grid(grid_n: int) -> Array:
    return build_grid([{"start": 0.0, "stop": 100.0, "num": grid_n}] * 2).reshape(-1, 2)


def gen_y_obs(rng: Array, s: Array, gt_ls: float) -> Array:
    rng_mu, rng_poiss = random.split(rng)
    K = matern_1_2(s, s, 1.0, gt_ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    return dist.Poisson(rate=jnp.exp(1.0 + mu)).sample(rng_poiss)


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple, obs_ratio: float = 0.15) -> Array:
    H, W = grid_shape
    total = H * W
    n_obs = int(obs_ratio * total)
    mask = jnp.zeros((H, W), dtype=bool)
    collected = 0
    while collected < n_obs:
        rng, rng_blob = random.split(rng)
        rngs = random.split(rng_blob, 4)
        cx = random.randint(rngs[0], (), 0, H)
        cy = random.randint(rngs[1], (), 0, W)
        rx = random.randint(rngs[2], (), H // 8, H // 4)
        ry = random.randint(rngs[3], (), W // 8, W // 4)
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        collected += int(jnp.sum(new_mask) - jnp.sum(mask))
        mask = new_mask
    if collected > n_obs:
        flat_idxs = jnp.argwhere(mask.flatten()).squeeze()
        rng_trim, _ = random.split(rngs[-1])
        selected = random.choice(rng_trim, flat_idxs, shape=(n_obs,), replace=False)
        return jnp.zeros(total, dtype=bool).at[selected].set(True)
    return mask.flatten()


def gen_train_dataloader(s: Array, priors: dict, batch_size: int = BATCH_SIZE):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, 1.0, ls)
            L = jnp.linalg.cholesky(K)
            yield {"s": s, "z": z, "conditionals": jnp.array([ls]), "f": f_jit(L, z)}

    return dataloader


# ---------------------------------------------------------------------------
# Inference model
# ---------------------------------------------------------------------------

def build_inference_model(s: Array, priors: dict) -> Callable:
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        ls = numpyro.sample("ls", priors["ls"])
        beta = numpyro.sample("beta", priors["beta"])
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            K = matern_1_2(s, s, 1.0, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.deterministic("mu", jnp.matmul(jnp.linalg.cholesky(K), z[0]))
        else:
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=jnp.exp(beta + mu)), obs=y)

    return poisson


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    train_step: Callable,
    valid_step: Callable,
    model: nn.Module,
    results_dir: Path,
    optimizer,
    train_num_steps: int = TRAIN_STEPS,
    valid_interval: int = VALID_INTERVAL,
    valid_steps: int = VALID_STEPS,
):
    flop_batch = next(loader(rng_train))
    rngs = {"params": rng_train, "extra": rng_test}
    kwargs = model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    state = TrainState.create(apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer)
    infer_flops, train_flops = estimate_flops(rng_train, state, train_step, flop_batch)
    parameters = nn.tabulate(model, rngs)(**flop_batch)
    parameters = int(
        parameters.split("Total Parameters: ")[-1].split(" ")[0].replace(",", "")
    )
    t0 = datetime.now()
    state = train(
        rng_train, model, optimizer, train_step, train_num_steps, loader,
        valid_step, valid_interval, valid_steps, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, valid_steps)["norm MSE"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder, infer_flops, train_flops, parameters


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def run_hmc(
    rng: Array,
    infer_model: Callable,
    y_obs: Array,
    obs_mask: Array,
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    nuts = NUTS(infer_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP)
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(infer_model, samples)(k2, surrogate_decoder=surrogate_decoder)
    post["infer_time"] = infer_time
    with open(results_dir / "hmc_samples.pkl", "wb") as f:
        pickle.dump({k: v for k, v in samples.items() if k in ["ls", "beta"]}, f)
    with open(results_dir / "hmc_pp.pkl", "wb") as f:
        pickle.dump(post, f)
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def collect_result(
    model_name, train_time, infer_time, eval_mse,
    infer_flops, train_flops, parameters,
    y_obs, post, obs_mask, samples, mcmc, seed, L,
):
    sq_res = (y_obs - post["obs"].mean(axis=0)) ** 2
    ess = az.ess(mcmc, method="mean") if mcmc is not None else {}
    ls_stats = numpyro_summary(mcmc.get_samples(group_by_chain=True), prob=0.9)["ls"] if mcmc is not None else {}
    res = {
        "model_name": model_name,
        "grid_size": L,
        "seed": seed,
        "train_time": train_time,
        "infer_time": infer_time,
        "total_time": (train_time or 0) + infer_time,
        "Test Norm MSE": eval_mse,
        "MSE(y, y_hat)": float(sq_res.mean()),
        "obs MSE(y, y_hat)": float(sq_res[obs_mask].mean()),
        "unobs MSE(y, y_hat)": float(sq_res[~obs_mask].mean()),
        "infer_flops": infer_flops,
        "train_flops": train_flops,
        "parameters": parameters,
        "num_chains": HMC_CHAINS,
        "inferred ls mean": float(samples["ls"].mean()) if "ls" in samples else None,
        "ls std": float(ls_stats["std"]) if ls_stats else None,
        "ls 5%": float(ls_stats["5.0%"]) if ls_stats else None,
        "ls 95%": float(ls_stats["95.0%"]) if ls_stats else None,
        "n_eff ls": float(ls_stats["n_eff"]) if ls_stats else None,
        "r_hat ls": float(ls_stats["r_hat"]) if ls_stats else None,
        "ESS ls": float(ess["ls"].mean().item()) if ess and "ls" in ess else None,
    }
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictive_means(grid_n, y_obs, y_hats, obs_mask, model_names, save_path):
    f_obs = y_obs.reshape(grid_n, grid_n)
    f_means_log = [jnp.log(y.mean(axis=0).reshape(grid_n, grid_n) + 1) for y in y_hats]
    f_obs_log = jnp.log(f_obs + 1)
    vmin = float(min(f.min() for f in f_means_log))
    vmax = float(max(f.max() for f in f_means_log))
    ncols = 2 + len(y_hats)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")
    masked = np.ma.masked_where(~obs_mask.reshape(grid_n, grid_n), f_obs_log)
    axes[0].imshow(masked, origin="lower", cmap=cmap)
    axes[0].set_title("observed y")
    axes[1].imshow(f_obs_log, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
    axes[1].set_title("true y")
    for ax, f_mean, name in zip(axes[2:], f_means_log, model_names):
        im = ax.imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
        ax.set_title(f"mean ŷ — {name}")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[-1])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def aggregate_and_plot(save_dir: Path):
    dfs = []
    for p in sorted(save_dir.glob("grid_*/res.csv")):
        dfs.append(pd.read_csv(p))
    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(save_dir / "aggregated_results.csv", index=False)

    models = df["model_name"].unique()
    metrics = ["MSE(y, y_hat)", "unobs MSE(y, y_hat)", "n_eff ls", "r_hat ls", "infer_time"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        for model in models:
            sub = df[df["model_name"] == model]
            if metric not in sub.columns or sub[metric].isnull().all():
                continue
            ax.plot(sub["grid_size"], sub[metric], marker="o", label=model)
        ax.set_title(metric)
        ax.set_xlabel("Grid size")
        ax.legend(fontsize=7)
    fig.savefig(save_dir / "scalability.png", dpi=150)
    plt.close(fig)
    print(f"Aggregated results → {save_dir / 'aggregated_results.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42, gt_ls: int = 10):
    rng = random.key(seed)
    save_dir = Path(f"results/poc_ls_{gt_ls}/")
    save_dir.mkdir(parents=True, exist_ok=True)

    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    cond_names = list(priors.keys())

    models = {
        "Baseline_GP": (None, None, None),
        "DeepRV + gMLP": (gMLPDeepRV(num_blks=2), deep_rv_train_step, deep_rv_valid_step),
        "FM-DeepRV (1 step)": (
            FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=1),
            flow_matching_train_step,
            flow_matching_valid_step,
        ),
        "FM-DeepRV (3 steps)": (
            FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=3),
            flow_matching_train_step,
            flow_matching_valid_step,
        ),
    }

    for grid_n in GRIDS:
        s = build_spatial_grid(grid_n)
        L = s.shape[0]
        grid_dir = save_dir / f"grid_{L}"
        grid_dir.mkdir(parents=True, exist_ok=True)

        rng, rng_obs, rng_mask, rng_train, rng_test, rng_infer = random.split(rng, 6)
        y_obs = gen_y_obs(rng_obs, s, gt_ls)
        obs_mask = gen_spatial_obs_mask(rng_mask, (grid_n, grid_n))
        infer_model = build_inference_model(s, priors)
        loader = gen_train_dataloader(s, priors)

        result, y_hats, all_samples = [], [], []

        for model_name, (nn_model, train_step, valid_step) in models.items():
            model_dir = grid_dir / model_name.replace(" ", "_").replace("(", "").replace(")", "")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Interruption-safe: reload if already done
            if (model_dir / "single_res.pkl").exists():
                print(f"  [{model_name} @ {grid_n}×{grid_n}] already done, loading.")
                with open(model_dir / "hmc_samples.pkl", "rb") as f:
                    samples = pickle.load(f)
                with open(model_dir / "hmc_pp.pkl", "rb") as f:
                    post = pickle.load(f)
                with open(model_dir / "single_res.pkl", "rb") as f:
                    res = pickle.load(f)
                y_hats.append(post["obs"])
                all_samples.append(samples)
                result.append(res)
                continue

            print(f"\n=== {model_name} | grid {grid_n}×{grid_n} | ls={gt_ls} ===")
            train_time = eval_mse = None
            infer_flops = train_flops = parameters = None
            surrogate_decoder = None
            mcmc = None

            if nn_model is not None:
                max_lr = MAX_LR
                lr_schedule = cosine_annealing_lr(TRAIN_STEPS, max_lr)
                optimizer = optax.chain(
                    optax.clip_by_global_norm(3.0),
                    optax.adamw(lr_schedule, weight_decay=1e-2),
                )
                wandb.init(
                    config={"model_name": model_name, "grid_size": L, "seed": seed},
                    mode="disabled",
                    reinit=True,
                )
                rng_train, rng_t = random.split(rng_train)
                rng_test, rng_v = random.split(rng_test)
                (
                    train_time, eval_mse, surrogate_decoder,
                    infer_flops, train_flops, parameters,
                ) = surrogate_model_train(
                    rng_t, rng_v, loader, train_step, valid_step,
                    nn_model, model_dir, optimizer,
                )

            rng_infer, rng_i = random.split(rng_infer)
            samples, mcmc, post, infer_time = run_hmc(
                rng_i, infer_model, y_obs, obs_mask, model_dir, surrogate_decoder
            )
            plot_infer_trace(
                {k: v for k, v in samples.items() if k in cond_names},
                mcmc, None, cond_names, model_dir / "infer_trace.png",
            )

            res = collect_result(
                model_name, train_time, infer_time, eval_mse,
                infer_flops, train_flops, parameters,
                y_obs, post, obs_mask,
                {k: v for k, v in samples.items() if k in cond_names},
                mcmc, seed, L,
            )
            with open(model_dir / "single_res.pkl", "wb") as f:
                pickle.dump(res, f)

            y_hats.append(post["obs"])
            all_samples.append({k: v for k, v in samples.items() if k in cond_names})
            result.append(res)

        # Wasserstein vs Baseline_GP
        baseline_samples = all_samples[0]
        for res, samp in zip(result, all_samples):
            for c in cond_names:
                if res["model_name"] == "Baseline_GP":
                    res[f"{c} wasserstein"] = float("nan")
                else:
                    res[f"{c} wasserstein"] = wasserstein_distance(
                        baseline_samples[c], samp[c]
                    )

        pd.DataFrame(result).to_csv(grid_dir / "res.csv", index=False)
        plot_predictive_means(
            grid_n, y_obs, y_hats, obs_mask,
            list(models.keys()), grid_dir / "predictive_means.png",
        )

    aggregate_and_plot(save_dir)


if __name__ == "__main__":
    main(seed=42, gt_ls=10)
    main(seed=57, gt_ls=20)
