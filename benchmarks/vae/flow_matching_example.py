#!/usr/bin/env python3
"""flow_matching_example.py

Compares DeepRV (gMLP) against FM-DeepRV with n_steps in {1, 3, 5, 10} on a
16×16 spatial grid with a Matérn-1/2 GP prior and Poisson likelihood.

FM-DeepRV is trained once; the same weights are reused for all n_steps
variants (n_steps only changes the decode-time ODE integration).

Run from the repo root:
    uv run python benchmarks/vae/flow_matching_example.py
"""

import sys

sys.path.append("benchmarks/vae")

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

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
from dl4bi_sps.kernels import matern_1_2
from dl4bi_sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, train
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

TRAIN_STEPS = 100_000
VALID_INTERVAL = 25_000
VALID_STEPS = 5_000
BATCH_SIZE = 32
MAX_LR = 1e-3
HMC_WARMUP = 1_000
HMC_SAMPLES = 1_000
HMC_CHAINS = 2
GRID_N = 16
FM_N_STEPS = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Valid step for gMLPDeepRV (FM uses flow_matching_valid_step)
# ---------------------------------------------------------------------------

@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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
            yield {
                "s": s,
                "z": z,
                "conditionals": jnp.array([ls]),
                "f": f_jit(L, z),
            }

    return dataloader


def gen_y_obs(rng: Array, s: Array, gt_ls: float) -> Array:
    rng_mu, rng_poiss = random.split(rng)
    K = matern_1_2(s, s, 1.0, gt_ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    return dist.Poisson(rate=jnp.exp(1.0 + mu)).sample(rng_poiss)


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple, obs_ratio: float = 0.7) -> Array:
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

def train_model(
    rng: Array,
    model_name: str,
    model,
    loader: Callable,
    train_step_fn: Callable,
    valid_step_fn: Callable,
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.adamw(cosine_annealing_lr(TRAIN_STEPS, MAX_LR), weight_decay=1e-2),
    )
    t0 = datetime.now()
    state = train(
        rng,
        model,
        optimizer,
        train_step_fn,
        TRAIN_STEPS,
        loader,
        valid_step_fn,
        VALID_INTERVAL,
        VALID_STEPS,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    print(f"  {model_name}: trained in {train_time:.0f}s")
    return state, train_time


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def run_hmc(
    rng: Array,
    infer_model: Callable,
    y_obs: Array,
    obs_mask: Array,
    surrogate_decoder: Optional[Callable] = None,
):
    nuts = NUTS(infer_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(
        nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP
    )
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(infer_model, samples)(k2, surrogate_decoder=surrogate_decoder)
    return samples, mcmc, post["obs"], infer_time


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def collect_result(model_name, train_time, infer_time, y_obs, y_hat, obs_mask, mcmc):
    sq_res = (y_obs - y_hat.mean(axis=0)) ** 2
    ls_stats = numpyro_summary(mcmc.get_samples(group_by_chain=True), prob=0.9)["ls"]
    return {
        "model": model_name,
        "train_time_s": round(train_time, 1),
        "infer_time_s": round(infer_time, 1),
        "MSE(y, y_hat)": float(sq_res.mean()),
        "obs MSE": float(sq_res[obs_mask].mean()),
        "unobs MSE": float(sq_res[~obs_mask].mean()),
        "ls mean": float(ls_stats["mean"]),
        "ls std": float(ls_stats["std"]),
        "ls 5%": float(ls_stats["5.0%"]),
        "ls 95%": float(ls_stats["95.0%"]),
        "n_eff": float(ls_stats["n_eff"]),
        "r_hat": float(ls_stats["r_hat"]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictive_means(
    grid_n: int,
    y_obs: Array,
    y_hats: list,
    obs_mask: Array,
    model_names: list,
    save_path: Path,
):
    f_obs = y_obs.reshape(grid_n, grid_n)
    f_means_log = [jnp.log(y.mean(axis=0).reshape(grid_n, grid_n) + 1) for y in y_hats]
    f_obs_log = jnp.log(f_obs + 1)

    vmin = float(jnp.min(jnp.array([f.min() for f in f_means_log])))
    vmax = float(jnp.max(jnp.array([f.max() for f in f_means_log])))

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 57, gt_ls: float = 20.0):
    rng = random.key(seed)
    rng_train, rng_infer, rng_mask, rng_obs, rng = random.split(rng, 5)

    wandb.init(mode="disabled")
    save_dir = Path("results/flow_matching_example/")
    save_dir.mkdir(parents=True, exist_ok=True)

    s = build_grid([{"start": 0.0, "stop": 100.0, "num": GRID_N}] * 2).reshape(-1, 2)
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}

    y_obs = gen_y_obs(rng_obs, s, gt_ls)
    obs_mask = gen_spatial_obs_mask(rng_mask, (GRID_N, GRID_N), obs_ratio=0.7)
    infer_model = build_inference_model(s, priors)
    loader = gen_train_dataloader(s, priors)

    results, y_hats, all_names = [], [], []

    # ------------------------------------------------------------------
    # DeepRV baseline
    # ------------------------------------------------------------------
    print("\n=== DeepRV (gMLP) ===")
    rng_train, rng_m = random.split(rng_train)
    rng_infer, rng_i = random.split(rng_infer)

    drv_model = gMLPDeepRV(num_blks=2)
    state_drv, train_time_drv = train_model(
        rng_m, "DeepRV", drv_model, loader, deep_rv_train_step, deep_rv_valid_step
    )
    decoder_drv = generate_surrogate_decoder(state_drv, drv_model)
    samples, mcmc, y_hat, infer_time = run_hmc(rng_i, infer_model, y_obs, obs_mask, decoder_drv)

    y_hats.append(y_hat)
    all_names.append("DeepRV")
    results.append(collect_result("DeepRV", train_time_drv, infer_time, y_obs, y_hat, obs_mask, mcmc))
    plot_infer_trace(samples, mcmc, None, list(priors.keys()), save_dir / "trace_DeepRV.png")

    # ------------------------------------------------------------------
    # FM-DeepRV: train once, evaluate with multiple n_steps
    # ------------------------------------------------------------------
    print("\n=== FM-DeepRV (training) ===")
    rng_train, rng_m = random.split(rng_train)

    fm_vf = FlowMatchingVectorField(num_blks=2)
    fm_base = FlowMatchingDeepRV(vf=fm_vf, n_steps=1)
    state_fm, train_time_fm = train_model(
        rng_m, "FM-DeepRV", fm_base, loader, flow_matching_train_step, flow_matching_valid_step
    )

    for k in FM_N_STEPS:
        model_name = f"FM-DeepRV ({k} step{'s' if k > 1 else ''})"
        print(f"\n=== {model_name} (HMC only) ===")
        rng_infer, rng_i = random.split(rng_infer)

        fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
        decoder_fm = generate_surrogate_decoder(state_fm, fm_k)
        samples, mcmc, y_hat, infer_time = run_hmc(
            rng_i, infer_model, y_obs, obs_mask, decoder_fm
        )

        y_hats.append(y_hat)
        all_names.append(model_name)
        results.append(
            collect_result(model_name, train_time_fm, infer_time, y_obs, y_hat, obs_mask, mcmc)
        )
        plot_infer_trace(
            samples, mcmc, None,
            list(priors.keys()),
            save_dir / f"trace_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)
    print("\n" + df.to_string(index=False))

    plot_predictive_means(
        GRID_N, y_obs, y_hats, obs_mask, all_names,
        save_dir / "predictive_means.png",
    )

    print(f"\nOutputs saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
