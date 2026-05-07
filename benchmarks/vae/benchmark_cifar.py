#!/usr/bin/env python3
"""benchmark_imagenet.py

Tests whether FM-DeepRV outperforms DeepRV on natural image patches (CIFAR-100),
and compares both surrogates against an exact GP baseline with a Matérn-3/2 kernel.

CIFAR-100 is used as the natural-image dataset: 100 classes, 32×32 colour images
converted to grayscale and resized to 16×16.  It downloads automatically via
tensorflow-datasets — no manual data preparation required.

Setup:
  - Source : z ~ N(0, I)   (256-dim for 16×16 grayscale patches)
  - Target : grayscale CIFAR-100 patch, normalised to [0, 1]
  - Loss   : OT-CFM for FM-DeepRV, MSE for DeepRV
  - Inference: observe 30% of pixels, Gaussian likelihood, HMC inpaints the rest

Models:
  - Exact GP (Matérn-3/2, non-centred parameterisation, infers log_ell + log_sigma)
  - DeepRV + gMLP
  - FM-DeepRV K=1,3,5

Run from the repo root:
    uv run python benchmarks/vae/benchmark_imagenet.py
"""

import os
import sys
sys.path.append("benchmarks/vae")

# Initialise JAX (and its CUDA context) before TF is imported.
import jax
import jax.numpy as jnp
from jax import Array, jit, random
jax.devices()  # force CUDA initialisation now

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

import pickle
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import flax.linen as nn
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer
from dl4bi_sps.kernels import matern_3_2
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

PATCH_SIZE = 16               # 16×16 grayscale patches (L=256, keeps GP tractable)
N_TRAIN = 50_000
N_TEST = 20
OBS_RATIO = 0.3               # fraction of pixels observed
OBS_NOISE = 0.05              # Gaussian likelihood sigma on [0,1] pixels
TRAIN_STEPS = 500_000
VALID_INTERVAL = 50_000
VALID_STEPS = 2_000
BATCH_SIZE = 64
MAX_LR = 1e-3
HMC_WARMUP = 1_000
HMC_SAMPLES = 1_000
HMC_CHAINS = 2
FM_K_STEPS = [1, 3, 5, 7]
N_BLOCKS = 4
GP_JITTER = 5e-4


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
# Data — ImageNet patches
# ---------------------------------------------------------------------------

def load_cifar100_patches(patch_size: int = PATCH_SIZE, n_train: int = N_TRAIN):
    """Load CIFAR-100, convert to grayscale, resize, normalise to [0, 1].

    Downloads automatically via tensorflow-datasets on first run (~160 MB).
    CIFAR-100 has 50k train and 10k test 32×32 colour images across 100 classes.
    """
    def preprocess(example):
        img = tf.cast(example["image"], tf.float32) / 255.0  # [32, 32, 3]
        img = tf.image.rgb_to_grayscale(img)                 # [32, 32, 1]
        img = tf.image.resize(img, [patch_size, patch_size]) # [P, P, 1]
        return tf.reshape(img, [-1])                         # [L]

    train_ds = (
        tfds.load("cifar100", split="train", as_supervised=False)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=50_000, seed=0)
        .take(n_train)
        .batch(n_train)
        .get_single_element()
        .numpy()
    )
    test_ds = (
        tfds.load("cifar100", split="test", as_supervised=False)
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
            f = patches[idx]
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            yield {
                "s": s,
                "z": z,
                "conditionals": jnp.array([0.0]),
                "f": f,
            }

    return dataloader


# ---------------------------------------------------------------------------
# Inference models
# ---------------------------------------------------------------------------

def build_gp_inpainting_model(s: Array) -> Callable:
    """Exact GP: non-centred parameterisation, infers log_ell + log_sigma.

    log_ell ~ N(log 0.3, 0.5)  — reasonable prior for [0,1]² pixel grid
    log_sigma ~ N(0, 1)
    z ~ N(0, I_L)
    f = chol(K(s,s; ell, sigma)) @ z
    y_obs ~ N(f[obs], obs_noise)
    """
    L = s.shape[0]

    def inpaint(obs_mask=None, y=None):
        log_ell = numpyro.sample("log_ell", dist.Normal(jnp.log(0.3), 0.5))
        log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 1.0))
        ell = jnp.exp(log_ell)
        sigma = jnp.exp(log_sigma)
        K = matern_3_2(s, s, sigma ** 2, ell) + GP_JITTER * jnp.eye(L)
        L_chol = jnp.linalg.cholesky(K)
        z = numpyro.sample("z", dist.Normal(jnp.zeros(L), 1.0))
        f = numpyro.deterministic("f", L_chol @ z)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(f, OBS_NOISE), obs=y)

    return inpaint


def build_surrogate_inpainting_model(s: Array) -> Callable:
    """Surrogate model: z ~ N(0,I), f = decoder(z), y_obs ~ N(f, sigma)."""
    surrogate_kwargs = {"s": s}

    def inpaint(surrogate_decoder=None, obs_mask=None, y=None):
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder is None:
            f = z[0]
        else:
            f = surrogate_decoder(
                z, jnp.array([0.0]), **surrogate_kwargs
            ).squeeze()
        numpyro.deterministic("f", f)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Normal(f, OBS_NOISE), obs=y)

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
) -> tuple:
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
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    return train_time, eval_mse, state, infer_flops, train_flops, parameters


def reload_state(ckpt_dir: Path, model: nn.Module, s: Array, optimizer) -> TrainState:
    """Restore weights from a saved checkpoint into a fresh TrainState."""
    L = s.shape[0]
    dummy_batch = {
        "s": s,
        "z": jnp.ones((1, L)),
        "conditionals": jnp.array([0.0]),
        "f": jnp.ones((1, L)),
    }
    rngs = {"params": random.key(0), "extra": random.key(1)}
    init_vars = model.init(rngs, **dummy_batch)
    init_params = init_vars.pop("params")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        state_template = TrainState.create(
            apply_fn=model.apply,
            params=init_params,
            kwargs=init_vars,
            tx=optimizer,
        )
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(ckpt_dir.absolute(), item={"state": state_template, "config": {}})
    return ckpt["state"]


# ---------------------------------------------------------------------------
# HMC inpainting
# ---------------------------------------------------------------------------

def run_hmc_surrogate(
    rng: Array,
    surrogate_model: Callable,
    y_obs: Array,
    obs_mask: Array,
    decoder: Callable,
):
    nuts = NUTS(surrogate_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(
        nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP
    )
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    samples = mcmc.get_samples()
    post = Predictive(surrogate_model, samples)(
        k2, surrogate_decoder=decoder, obs_mask=obs_mask
    )
    return samples, mcmc, post, infer_time


def run_hmc_gp(
    rng: Array,
    gp_model: Callable,
    y_obs: Array,
    obs_mask: Array,
):
    nuts = NUTS(gp_model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(
        nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP
    )
    t0 = datetime.now()
    mcmc.run(k1, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    samples = mcmc.get_samples()
    post = Predictive(gp_model, samples)(k2, obs_mask=obs_mask)
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mean_ess_z(mcmc) -> float:
    ess = az.ess(mcmc, method="mean", var_names=["z"])
    return float(ess["z"].values.mean())


def mean_rhat_z(samples_by_chain: dict) -> float:
    idata = az.convert_to_inference_data(
        {k: np.array(v) for k, v in samples_by_chain.items()}
    )
    rhat = az.rhat(idata, var_names=["z"])
    return float(rhat["z"].values.mean())


def ess_log_ell(mcmc) -> float:
    ess = az.ess(mcmc, method="mean", var_names=["log_ell"])
    return float(ess["log_ell"].values)


def rhat_log_ell(samples_by_chain: dict) -> float:
    idata = az.convert_to_inference_data(
        {k: np.array(v) for k, v in samples_by_chain.items()}
    )
    rhat = az.rhat(idata, var_names=["log_ell"])
    return float(rhat["log_ell"].values)


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

    col_titles = ["true", f"masked ({int(OBS_RATIO*100)}%)"] + model_names
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9)

    for i in range(n_imgs):
        true = np.array(true_imgs[i]).reshape(patch_size, patch_size)
        mask_2d = np.array(obs_masks[i]).reshape(patch_size, patch_size)
        masked = np.ma.masked_where(~mask_2d, true)

        axes[i, 0].imshow(true, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(masked, cmap="gray", vmin=0, vmax=1)
        for j, recon in enumerate(recon_means[i]):
            axes[i, 2 + j].imshow(
                np.array(recon).reshape(patch_size, patch_size),
                cmap="gray", vmin=0, vmax=1,
            )

    for ax in axes.flatten():
        ax.axis("off")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

GP_SCALAR_KEYS = {
    "model_name", "img_idx", "infer_time",
    "MSE (all)", "MSE (obs)", "MSE (unobs)",
    "mean ESS z", "mean r_hat z",
    "ESS log_ell", "r_hat log_ell", "mean log_ell",
}

SURROGATE_SCALAR_KEYS = {
    "model_name", "img_idx", "infer_time",
    "MSE (all)", "MSE (obs)", "MSE (unobs)",
    "mean ESS z", "mean r_hat z",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 42):
    rng = random.key(seed)
    save_dir = Path("results/cifar100_benchmark/").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-100 patches...")
    train_patches, val_patches = load_cifar100_patches()
    s = build_pixel_grid()
    L = s.shape[0]
    print(f"  Train: {train_patches.shape}, Val: {val_patches.shape}, L={L}")

    # --- Train surrogates ---
    train_configs = {
        "DeepRV + gMLP": (
            gMLPDeepRV(num_blks=N_BLOCKS),
            deep_rv_train_step,
            deep_rv_valid_step,
        ),
        "FM-DeepRV": (
            FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=N_BLOCKS), n_steps=1),
            flow_matching_train_step,
            flow_matching_valid_step,
        ),
    }

    trained_states = {}
    for model_name, (nn_model, train_step, valid_step) in train_configs.items():
        model_dir = (save_dir / model_name.replace(" ", "_").replace("+", "plus")).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = model_dir / "model.ckpt"

        lr_schedule = cosine_annealing_lr(TRAIN_STEPS, MAX_LR)
        optimizer = optax.chain(
            optax.clip_by_global_norm(3.0),
            optax.adamw(lr_schedule, weight_decay=1e-2),
        )

        if ckpt_dir.exists():
            print(f"  [{model_name}] checkpoint found, reloading.")
            state = reload_state(ckpt_dir, nn_model, s, optimizer)
        else:
            print(f"\n=== Training {model_name} ===")
            rng, rng_t, rng_v = random.split(rng, 3)
            loader = gen_train_dataloader(train_patches, s)
            wandb.init(
                config={"model_name": model_name, "dataset": "cifar100", "seed": seed},
                mode="disabled", reinit=True,
            )
            train_time, eval_mse, state, _, _, _ = surrogate_model_train(
                rng_t, rng_v, loader, train_step, valid_step,
                nn_model, model_dir, optimizer,
            )
            print(f"  trained in {train_time:.0f}s  |  eval norm MSE: {eval_mse:.4f}")

        trained_states[model_name] = (state, nn_model)

    # --- Build eval decoders ---
    state_drv, drv_model = trained_states["DeepRV + gMLP"]
    state_fm, fm_base_model = trained_states["FM-DeepRV"]
    fm_vf = fm_base_model.vf

    surrogate_decoders = {
        "DeepRV + gMLP": generate_surrogate_decoder(state_drv, drv_model),
    }
    for k in FM_K_STEPS:
        fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
        surrogate_decoders[f"FM-DeepRV (K={k})"] = generate_surrogate_decoder(state_fm, fm_k)

    gp_model = build_gp_inpainting_model(s)
    surrogate_model = build_surrogate_inpainting_model(s)

    # --- Test images ---
    rng, rng_test = random.split(rng)
    test_idxs = random.choice(rng_test, val_patches.shape[0], shape=(N_TEST,), replace=False)
    test_imgs = val_patches[test_idxs]

    results = []
    vis_true, vis_masked, vis_masks = [], [], []
    vis_recons = []  # one list-of-recons per image, ordered: GP then surrogates

    all_model_names = ["GP (Matérn-3/2)"] + list(surrogate_decoders.keys())

    for img_i in range(N_TEST):
        true_f = test_imgs[img_i]
        rng, rng_mask, rng_noise = random.split(rng, 3)
        obs_mask = gen_obs_mask(rng_mask, L)
        y_obs = jnp.where(
            obs_mask,
            true_f + OBS_NOISE * random.normal(rng_noise, (L,)),
            jnp.zeros(L),
        )

        vis_true.append(true_f)
        vis_masked.append(true_f * obs_mask)
        vis_masks.append(obs_mask)
        img_recons = []

        # --- GP baseline ---
        gp_cache = save_dir / f"img{img_i}_GP.pkl"
        if gp_cache.exists():
            print(f"  [GP | img {img_i}] cached, loading.")
            with open(gp_cache, "rb") as fh:
                gp_res = pickle.load(fh)
        else:
            print(f"\n=== GP baseline | test image {img_i+1}/{N_TEST} ===")
            rng, rng_i = random.split(rng)
            gp_samples, gp_mcmc, gp_post, gp_time = run_hmc_gp(
                rng_i, gp_model, y_obs, obs_mask
            )
            f_mean = gp_post["f"].mean(axis=0)
            mse_all, mse_obs, mse_unobs = image_mse(true_f, f_mean, obs_mask)
            gp_sbc = {k: np.array(v) for k, v in gp_mcmc.get_samples(group_by_chain=True).items()}
            gp_res = {
                "model_name": "GP (Matérn-3/2)",
                "img_idx": int(img_i),
                "infer_time": gp_time,
                "MSE (all)": mse_all,
                "MSE (obs)": mse_obs,
                "MSE (unobs)": mse_unobs,
                "mean ESS z": mean_ess_z(gp_mcmc),
                "mean r_hat z": mean_rhat_z(gp_sbc),
                "ESS log_ell": ess_log_ell(gp_mcmc),
                "r_hat log_ell": rhat_log_ell(gp_sbc),
                "mean log_ell": float(gp_samples["log_ell"].mean()),
                "f_mean": np.array(f_mean),
                "samples_by_chain": gp_sbc,
            }
            with open(gp_cache, "wb") as fh:
                pickle.dump(gp_res, fh)

        results.append({k: gp_res.get(k, float("nan")) for k in GP_SCALAR_KEYS})
        img_recons.append(jnp.array(gp_res["f_mean"]))

        # --- Surrogates ---
        for model_name, decoder in surrogate_decoders.items():
            safe_key = f"img{img_i}_{model_name}".replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
            cache_path = save_dir / f"{safe_key}.pkl"

            if cache_path.exists():
                print(f"  [{model_name} | img {img_i}] cached, loading.")
                with open(cache_path, "rb") as fh:
                    res = pickle.load(fh)
            else:
                print(f"\n=== {model_name} | test image {img_i+1}/{N_TEST} ===")
                rng, rng_i = random.split(rng)
                samples, mcmc, post, infer_time = run_hmc_surrogate(
                    rng_i, surrogate_model, y_obs, obs_mask, decoder
                )
                f_mean = post["f"].mean(axis=0)
                mse_all, mse_obs, mse_unobs = image_mse(true_f, f_mean, obs_mask)
                sbc = {k: np.array(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
                res = {
                    "model_name": model_name,
                    "img_idx": int(img_i),
                    "infer_time": infer_time,
                    "MSE (all)": mse_all,
                    "MSE (obs)": mse_obs,
                    "MSE (unobs)": mse_unobs,
                    "mean ESS z": mean_ess_z(mcmc),
                    "mean r_hat z": mean_rhat_z(sbc),
                    "f_mean": np.array(f_mean),
                    "samples_by_chain": sbc,
                }
                with open(cache_path, "wb") as fh:
                    pickle.dump(res, fh)

            results.append({k: res.get(k, float("nan")) for k in SURROGATE_SCALAR_KEYS})
            img_recons.append(jnp.array(res["f_mean"]))

        vis_recons.append(img_recons)

    # --- Aggregate and save ---
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)
    summary_cols = [
        c for c in ["MSE (all)", "MSE (unobs)", "mean ESS z", "mean r_hat z",
                     "ESS log_ell", "r_hat log_ell", "mean log_ell", "infer_time"]
        if c in df.columns
    ]
    summary = df.groupby("model_name")[summary_cols].mean()
    print("\n" + summary.to_string())
    summary.to_csv(save_dir / "summary.csv")

    n_vis = min(5, N_TEST)
    plot_reconstructions(
        PATCH_SIZE,
        vis_true[:n_vis],
        vis_masked[:n_vis],
        vis_recons[:n_vis],
        all_model_names,
        vis_masks[:n_vis],
        save_dir / "reconstructions.png",
    )
    print(f"\nOutputs saved to {save_dir}")


if __name__ == "__main__":
    main()
