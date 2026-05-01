#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF CUDA warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

"""benchmark_mnist.py

Tests the hypothesis that FM-DeepRV outperforms DeepRV on complex image
distributions, where a single forward pass is insufficient.

Setup:
  - Source : z ~ N(0, I)   (256-dim for 16×16 patches)
  - Target : f = normalised MNIST patch  (no explicit Cholesky needed)
  - Loss   : OT-CFM for FM-DeepRV, MSE for DeepRV
  - Inference: observe 30% of pixels, HMC inpaints the rest

Models:
  - DeepRV + gMLP      (single forward pass baseline)
  - FM-DeepRV (K=1)
  - FM-DeepRV (K=3)
  - FM-DeepRV (K=5)

Key claim: FM-DeepRV generalises beyond GP priors because it does not require
an explicit Cholesky. The iterative ODE is essential on complex distributions.

Run from the repo root:
    python benchmarks/vae/benchmark_mnist.py
"""

import sys

sys.path.append("benchmarks/vae")

import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")   # TF is CPU-only; JAX owns the GPU
import tensorflow_datasets as tfds
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from dl4bi_sps.utils import build_grid

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

PATCH_SIZE = 16               # resize MNIST 28×28 → 16×16
N_TRAIN = 50_000              # MNIST training images to use
N_TEST = 20                   # test images to run HMC on
OBS_RATIO = 0.3               # fraction of pixels observed
OBS_NOISE = 0.05              # Gaussian likelihood sigma on [0,1] pixels
TRAIN_STEPS = 100_000
VALID_INTERVAL = 25_000
VALID_STEPS = 2_000
BATCH_SIZE = 64
MAX_LR = 1e-3
HMC_WARMUP = 1_000
HMC_SAMPLES = 1_000
HMC_CHAINS = 2
FM_K_STEPS = [1, 3, 5]


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
# Data — MNIST patches
# ---------------------------------------------------------------------------

def load_mnist_patches(patch_size: int = PATCH_SIZE, n_train: int = N_TRAIN):
    """Load MNIST, resize to patch_size×patch_size, normalise to [0,1]."""
    import tensorflow as tf

    def preprocess(example):
        img = tf.cast(example["image"], tf.float32) / 255.0
        img = tf.image.resize(img, [patch_size, patch_size])
        return tf.reshape(img, [-1])  # flatten to [patch_size²]

    train_ds = (
        tfds.load("mnist", split="train", as_supervised=False)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .take(n_train)
        .batch(n_train)
        .get_single_element()
        .numpy()
    )
    test_ds = (
        tfds.load("mnist", split="test", as_supervised=False)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(10_000)
        .get_single_element()
        .numpy()
    )
    return jnp.array(train_ds), jnp.array(test_ds)


def build_pixel_grid(patch_size: int = PATCH_SIZE) -> Array:
    """2D pixel coordinates in [0,1]×[0,1], shape [patch_size², 2]."""
    return build_grid(
        [{"start": 0.0, "stop": 1.0, "num": patch_size}] * 2
    ).reshape(-1, 2)


def gen_obs_mask(rng: Array, n_pixels: int, obs_ratio: float = OBS_RATIO) -> Array:
    n_obs = int(obs_ratio * n_pixels)
    idx = random.permutation(rng, n_pixels)[:n_obs]
    return jnp.zeros(n_pixels, dtype=bool).at[idx].set(True)


def gen_train_dataloader(patches: Array, s: Array, batch_size: int = BATCH_SIZE):
    N = patches.shape[0]

    def dataloader(rng_data):
        while True:
            rng_data, rng_idx, rng_z = random.split(rng_data, 3)
            idx = random.choice(rng_idx, N, shape=(batch_size,), replace=False)
            f = patches[idx]                                             # [B, L]
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            yield {
                "s": s,
                "z": z,
                "conditionals": jnp.array([0.0]),  # dummy — no hyperparams
                "f": f,
            }

    return dataloader


# ---------------------------------------------------------------------------
# Inference model — Gaussian inpainting
# ---------------------------------------------------------------------------

def build_inpainting_model(s: Array, obs_mask: Array) -> Callable:
    """HMC model: z ~ N(0,I), f = surrogate(z), y_obs ~ N(f[mask], sigma)."""
    surrogate_kwargs = {"s": s}

    def inpaint(surrogate_decoder=None, y=None):
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            # trivial baseline: identity (f = z)
            f = z[0]
        else:
            f = surrogate_decoder(
                z, jnp.array([0.0]), **surrogate_kwargs
            ).squeeze()
        numpyro.deterministic("f", f)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample(
                "obs", dist.Normal(f, OBS_NOISE), obs=y
            )

    return inpaint


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
        rng_train, model, optimizer, train_step, TRAIN_STEPS, loader,
        valid_step, VALID_INTERVAL, VALID_STEPS, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, VALID_STEPS)["norm MSE"]
    save_ckpt(state, DictConfig({}), (results_dir / "model.ckpt").resolve())
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder, infer_flops, train_flops, parameters


# ---------------------------------------------------------------------------
# HMC inpainting
# ---------------------------------------------------------------------------

def run_hmc_inpaint(
    rng: Array,
    inpaint_model: Callable,
    y_obs_partial: Array,
    obs_mask: Array,
    surrogate_decoder: Optional[Callable] = None,
):
    nuts = NUTS(inpaint_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(
        nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP
    )
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, y=y_obs_partial)
    infer_time = (datetime.now() - t0).total_seconds()
    samples = mcmc.get_samples()
    post = Predictive(inpaint_model, samples)(k2, surrogate_decoder=surrogate_decoder)
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_ess_z(mcmc) -> float:
    """Mean ESS across all z dimensions."""
    ess = az.ess(mcmc, method="mean", var_names=["z"])
    return float(ess["z"].values.mean())


def image_mse(true_f, post_f_mean, obs_mask):
    sq = (true_f - post_f_mean) ** 2
    return float(sq.mean()), float(sq[obs_mask].mean()), float(sq[~obs_mask].mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstructions(
    patch_size, true_imgs, masked_imgs, recon_means, model_names, obs_masks, save_path
):
    n_imgs = len(true_imgs)
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(
        n_imgs, n_cols, figsize=(3 * n_cols, 3 * n_imgs), constrained_layout=True
    )
    if n_imgs == 1:
        axes = axes[None]

    col_titles = ["true", "masked (30%)"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i in range(n_imgs):
        true = np.array(true_imgs[i]).reshape(patch_size, patch_size)
        masked = np.ma.masked_where(~np.array(obs_masks[i]).reshape(patch_size, patch_size), true)

        axes[i, 0].imshow(true, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(masked, cmap="gray", vmin=0, vmax=1)
        for j, recon in enumerate(recon_means[i]):
            axes[i, 2 + j].imshow(
                np.array(recon).reshape(patch_size, patch_size), cmap="gray", vmin=0, vmax=1
            )

    for ax in axes.flatten():
        ax.axis("off")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42):
    rng = random.key(seed)
    save_dir = Path("results/mnist_benchmark/")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST...")
    train_patches, test_patches = load_mnist_patches()
    s = build_pixel_grid()
    L = s.shape[0]
    print(f"  Train: {train_patches.shape}, Test: {test_patches.shape}, L={L}")

    # --- Models to train ---
    # FM-DeepRV: train once, decode with multiple K at inference
    train_configs = {
        "DeepRV + gMLP": (
            gMLPDeepRV(num_blks=2),
            deep_rv_train_step,
            deep_rv_valid_step,
        ),
        "FM-DeepRV": (
            FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=1),
            flow_matching_train_step,
            flow_matching_valid_step,
        ),
    }

    trained_states = {}

    for model_name, (nn_model, train_step, valid_step) in train_configs.items():
        model_dir = save_dir / model_name.replace(" ", "_").replace("+", "plus")
        model_dir.mkdir(parents=True, exist_ok=True)

        if (model_dir / "model.ckpt").exists():
            print(f"  [{model_name}] checkpoint found, skipping training.")
            # reload state for decoder generation
            from orbax.checkpoint import PyTreeCheckpointer
            ckptr = PyTreeCheckpointer()
            ckpt = ckptr.restore(model_dir / "model.ckpt")
            rng, rng_t, rng_v = random.split(rng, 3)
            loader = gen_train_dataloader(train_patches, s)
            flop_batch = next(loader(rng_t))
            rngs = {"params": rng_t, "extra": rng_v}
            kwargs = nn_model.init(rngs, **flop_batch)
            params = kwargs.pop("params")
            lr_schedule = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
            optimizer = optax.chain(
                optax.clip_by_global_norm(3.0),
                optax.adamw(lr_schedule, weight_decay=1e-2),
            )
            state = TrainState.create(
                apply_fn=nn_model.apply,
                params=ckpt["state"]["params"],
                kwargs=ckpt["state"]["kwargs"],
                tx=optimizer,
            )
            trained_states[model_name] = (state, nn_model)
            continue

        print(f"\n=== Training {model_name} ===")
        rng, rng_t, rng_v = random.split(rng, 3)
        loader = gen_train_dataloader(train_patches, s)
        lr_schedule = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
        optimizer = optax.chain(
            optax.clip_by_global_norm(3.0),
            optax.adamw(lr_schedule, weight_decay=1e-2),
        )
        wandb.init(
            config={"model_name": model_name, "dataset": "mnist", "seed": seed},
            mode="disabled", reinit=True,
        )
        train_time, eval_mse, _, infer_flops, train_flops, parameters = surrogate_model_train(
            rng_t, rng_v, loader, train_step, valid_step, nn_model, model_dir, optimizer
        )
        print(f"  trained in {train_time:.0f}s  |  eval norm MSE: {eval_mse:.4f}")

        # reload from checkpoint to get clean state
        from orbax.checkpoint import PyTreeCheckpointer
        ckptr = PyTreeCheckpointer()
        ckpt = ckptr.restore(model_dir / "model.ckpt")
        state = TrainState.create(
            apply_fn=nn_model.apply,
            params=ckpt["state"]["params"],
            kwargs=ckpt["state"]["kwargs"],
            tx=optimizer,
        )
        trained_states[model_name] = (state, nn_model)

    # --- Build all eval decoders ---
    # DeepRV: one decoder
    # FM-DeepRV: one decoder per K value (shared weights)
    state_drv, drv_model = trained_states["DeepRV + gMLP"]
    state_fm, fm_base_model = trained_states["FM-DeepRV"]
    fm_vf = fm_base_model.vf

    eval_decoders = {
        "DeepRV + gMLP": generate_surrogate_decoder(state_drv, drv_model),
    }
    for k in FM_K_STEPS:
        name = f"FM-DeepRV (K={k})"
        fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
        eval_decoders[name] = generate_surrogate_decoder(state_fm, fm_k)

    eval_model_names = list(eval_decoders.keys())

    # --- Inference on test images ---
    rng, rng_test = random.split(rng)
    test_idxs = random.choice(rng_test, test_patches.shape[0], shape=(N_TEST,), replace=False)
    test_imgs = test_patches[test_idxs]

    results = []
    vis_true, vis_masked, vis_recons, vis_masks = [], [], [], []

    for img_i in range(N_TEST):
        true_f = test_imgs[img_i]  # [L]
        rng, rng_mask, rng_noise = random.split(rng, 3)
        obs_mask = gen_obs_mask(rng_mask, L)
        y_partial = true_f[obs_mask] + OBS_NOISE * random.normal(rng_noise, (int(obs_mask.sum()),))
        inpaint_model = build_inpainting_model(s, obs_mask)

        vis_true.append(true_f)
        vis_masked.append(true_f * obs_mask)
        vis_masks.append(obs_mask)
        img_recons = []

        for model_name, decoder in eval_decoders.items():
            result_key = f"img{img_i}_{model_name}"
            cache_path = save_dir / f"{result_key.replace(' ', '_').replace('=', '')}.pkl"

            if cache_path.exists():
                print(f"  [{model_name} | img {img_i}] cached, loading.")
                with open(cache_path, "rb") as f:
                    res = pickle.load(f)
            else:
                print(f"\n=== {model_name} | test image {img_i+1}/{N_TEST} ===")
                rng, rng_i = random.split(rng)
                samples, mcmc, post, infer_time = run_hmc_inpaint(
                    rng_i, inpaint_model, y_partial, obs_mask, decoder
                )
                f_mean = post["f"].mean(axis=0)
                mse_all, mse_obs, mse_unobs = image_mse(true_f, f_mean, obs_mask)
                ess = mean_ess_z(mcmc)
                res = {
                    "model_name": model_name,
                    "img_idx": int(img_i),
                    "infer_time": infer_time,
                    "MSE (all)": mse_all,
                    "MSE (obs)": mse_obs,
                    "MSE (unobs)": mse_unobs,
                    "mean ESS z": ess,
                    "f_mean": np.array(f_mean),
                }
                with open(cache_path, "wb") as f:
                    pickle.dump(res, f)

            results.append({k: v for k, v in res.items() if k != "f_mean"})
            img_recons.append(jnp.array(res["f_mean"]))

        vis_recons.append(img_recons)

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)

    summary = df.groupby("model_name")[["MSE (all)", "MSE (unobs)", "mean ESS z", "infer_time"]].mean()
    print("\n" + summary.to_string())
    summary.to_csv(save_dir / "summary.csv")

    # --- Visualise a few reconstructions ---
    n_vis = min(5, N_TEST)
    plot_reconstructions(
        PATCH_SIZE,
        vis_true[:n_vis],
        vis_masked[:n_vis],
        vis_recons[:n_vis],
        eval_model_names,
        vis_masks[:n_vis],
        save_dir / "reconstructions.png",
    )

    print(f"\nOutputs saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()
