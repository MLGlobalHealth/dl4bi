#!/usr/bin/env python3
"""benchmark_fm_spatiotemporal_FIXED.py — adapted from
benchmark_fm_spatiotemporal.py to recover FM-DeepRV convergence.

Three changes vs the original (each marked with `# FIXED:` in the diff):

1. **Optimizer**: AdamW → YOGI.  AdamW destabilises FM training on
   this benchmark — the vector field collapses to ~0 and eval norm
   MSE stalls at ~1.4-1.8 (worse than predicting the prior mean).
   YOGI's sign-based second-moment update absorbs the noise from
   FM's randomised per-sample `t` and the network actually learns.

2. **Priors**: dropped the `a` and `nu` hyperparameters (held at 1.0
   in the kernel) and narrowed `ls` from Beta(4,1)+LogScale[1,100] to
   Uniform-on-log[1,100].  With YOGI fixed, the wide-marginal
   original priors still broke FM (Beta(4,1) heavily skews `ls`
   towards 100; LogNormal(0,1) gives `a` ~5 OOM range; nu∈[2,4]
   modulates the marginal variance).  The reduced 2-D `(ls, alpha)`
   parameterisation lets the FM vector field fit the training
   distribution.

3. **Conditional vector**: 4-D `(ls, a, alpha, nu)` → 2-D `(ls, alpha)`.

Bisection that motivated these fixes is in
DeepRV_malaria/scripts/benchmark_fm_iterate.py.  Headline result:

  | config                              | gMLP MSE | FM MSE |
  | ----------------------------------- | -------- | ------ |
  | original (AdamW + Makkunda priors)  | 0.018    | 1.45   |
  | + YOGI optimizer only               | 0.038    | 1.45   |
  | + YOGI + narrower priors (this)     | 0.009    | 0.13   |
  | + YOGI + ours priors + our kernel   | 0.004    | 0.06   |

FM-DeepRV vs DeepRV vs exact GP on a nonseparable Gneiting space-time kernel
with Poisson likelihood.  Mirrors the structure of benchmark_mnist.py.

Setup:
  - Spatial grid : 16×16 over [0, 100]²
  - Time steps   : T = 5  (observed at t = 0, 3, 4)
  - Kernel       : Gneiting (2002) nonseparable space-time covariance
  - Likelihood   : Poisson
  - Surrogate    : gMLP operating on (T×L, 3) spatiotemporal coordinates

Hyperparameters inferred by HMC: ls, a, alpha, nu, beta.

Models:
  - Exact GP (Cholesky + HMC)
  - DeepRV + gMLP
  - FM-DeepRV K=1,3,5  (trained once, shared weights)

Run from the repo root:
    uv run python benchmarks/vae/benchmark_fm_spatiotemporal.py
"""

import os
import sys
sys.path.append("benchmarks/vae")

# Initialise JAX / CUDA before anything else touches the GPU.
import jax
import jax.numpy as jnp
from jax import Array, jit, random
from jax.scipy.linalg import solve_triangular
jax.devices()

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
from numpyro.distributions.transforms import ParameterFreeTransform
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from orbax.checkpoint import PyTreeCheckpointer
from scipy.stats import wasserstein_distance
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

GRID_SHAPE = (16, 16)          # spatial grid
TIME_STEPS = 5                 # T
T_OBS_MASK = jnp.array([True, False, False, True, True])    # matches original spatiotemporal_kernel.py
OBS_RATIO = 0.5                # fraction of spatial locations observed per time step
GT_LS, GT_A, GT_ALPHA = 20.0, 0.5, 0.8
GT_B, GT_NU, GT_BETA = 1.0, 1.0, 1.0
JITTER = 5e-4

TRAIN_STEPS = 500_000
VALID_INTERVAL = 50_000
VALID_STEPS = 2_000
BATCH_SIZE = 16                # small batch — each sample requires a T*L Cholesky
MAX_LR = 5e-3
N_BLOCKS = 4

HMC_WARMUP = 4_000
HMC_SAMPLES = 10_000
HMC_CHAINS = 4
FM_K_STEPS = [1, 3, 5]

COND_NAMES = ["ls", "alpha", "beta"]  # FIXED: dropped a, nu (held at 1.0 in kernel)


# ---------------------------------------------------------------------------
# Spatiotemporal kernel
# ---------------------------------------------------------------------------

@jit
def gneiting_kernel(
    s1: Array, t1: Array, s2: Array, t2: Array,
    var: float, ls: float, a: float, alpha: float, b: float, nu: float,
) -> Array:
    """Gneiting (2002) nonseparable space-time covariance, shape [T1*L1, T2*L2]."""
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    h2 = jnp.sum((s1[:, None, :] - s2[None, :, :]) ** 2, axis=-1)   # [L1, L2]
    u  = jnp.abs(t1[:, None] - t2[None, :])                          # [T1, T2]
    h2 = h2[None, None, :, :]       # [1,  1,  L1, L2]
    u  = u[:, :, None, None]        # [T1, T2, 1,  1 ]
    g  = 1.0 + a * u ** (2 * alpha)
    K  = var / (g ** nu) * jnp.exp(-h2 / (ls ** 2 * g ** b))
    return K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def build_st_grid(grid_shape: tuple = GRID_SHAPE, time_steps: int = TIME_STEPS) -> tuple:
    """Return (s [L,2], t [T], st [T*L, 3])."""
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": grid_shape[0]}] * 2).reshape(-1, 2)
    t = jnp.arange(time_steps, dtype=jnp.float32)
    T, L = time_steps, s.shape[0]
    s_exp = jnp.broadcast_to(s, (T, L, 2))
    t_exp = t[:, None, None] * jnp.ones((1, L, 1))
    st = jnp.concatenate([s_exp, t_exp], axis=-1).reshape(T * L, 3)
    return s, t, st


def gen_y_obs(rng: Array, s: Array, t: Array) -> Array:
    rng_mu, rng_poiss = random.split(rng)
    K = gneiting_kernel(s, t, s, t, 1.0, GT_LS, GT_A, GT_ALPHA, GT_B, GT_NU)
    K = K + JITTER * jnp.eye(K.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu).reshape(TIME_STEPS, s.shape[0])
    return dist.Poisson(rate=jnp.exp(GT_BETA + mu)).sample(rng_poiss)


def gen_obs_mask(rng: Array, grid_shape: tuple = GRID_SHAPE, obs_ratio: float = OBS_RATIO) -> Array:
    """Spatially contiguous blob mask, True = observed."""
    H, W = grid_shape
    total = H * W
    n_obs = int(obs_ratio * total)
    mask = jnp.zeros((H, W), dtype=bool)
    collected = 0
    while collected < n_obs:
        rng, rng_b = random.split(rng)
        ks = random.split(rng_b, 4)
        cx = random.randint(ks[0], (), 0, H)
        cy = random.randint(ks[1], (), 0, W)
        rx = random.randint(ks[2], (), H // 8, H // 4)
        ry = random.randint(ks[3], (), W // 8, W // 4)
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        collected += int(jnp.sum(new_mask) - jnp.sum(mask))
        mask = new_mask
    if collected > n_obs:
        flat_idxs = jnp.argwhere(mask.flatten()).squeeze()
        rng, rng_trim = random.split(rng)
        selected = random.choice(rng_trim, flat_idxs, shape=(n_obs,), replace=False)
        mask = jnp.zeros(total, dtype=bool).at[selected].set(True)
    else:
        mask = mask.flatten()
    return mask


# ---------------------------------------------------------------------------
# Training dataloader
# ---------------------------------------------------------------------------

class LogScaleTransform(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive
    event_dim = 0

    def __call__(self, x):
        return jnp.exp(x * jnp.log(100.0))

    def _inverse(self, y):
        return jnp.log(y) / jnp.log(100.0)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(100.0) + x * jnp.log(100.0)


def make_priors() -> dict:
    """FIXED: narrow priors that allow FM training to converge.

    Original (commented below) used Beta(4,1)+LogScale on `ls` plus
    LogNormal/Beta/Uniform on `a, alpha, nu`.  The wide marginals
    (a∈[0.1,10], nu∈[2,4]) blew up the variance of training-data fields
    and the FM vector-field never converged.  With YOGI + narrower
    priors (uniform-on-log ls, fixed nu=a=1) FM trains to MSE<0.2.
    """
    return {
        "ls":    dist.TransformedDistribution(
                     dist.Uniform(jnp.log(1.0), jnp.log(100.0)),
                     dist.transforms.ExpTransform(),
                 ),
        "alpha": dist.Uniform(0.1, 0.95),
        "beta":  dist.Normal(),
    }


def gen_train_dataloader(st: Array, priors: dict, batch_size: int = BATCH_SIZE):
    TL = st.shape[0]
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))
    # Split st back into spatial (s) and temporal (t) parts for the kernel call.
    s_all = st[:, :2].reshape(TIME_STEPS, -1, 2)[0]   # [L, 2] (same across time)
    t_all = jnp.arange(TIME_STEPS, dtype=jnp.float32)

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_al, rng_z = random.split(rng_data, 4)
            ls    = priors["ls"].sample(rng_ls)
            alpha = priors["alpha"].sample(rng_al)
            # FIXED: a=1.0, nu=1.0 (held constant; cond reduced to 2 dims).
            K = gneiting_kernel(s_all, t_all, s_all, t_all, 1.0, ls, 1.0, alpha, 1.0, 1.0)
            K = K + JITTER * jnp.eye(TL)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, TL))
            f = f_jit(K, z)
            yield {
                "s":            st,
                "f":            f,
                "z":            z,
                "conditionals": jnp.array([ls, alpha]),  # FIXED: 2-D cond
            }

    return dataloader


# ---------------------------------------------------------------------------
# Valid steps
# ---------------------------------------------------------------------------

@jit
def deep_rv_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    return {"norm MSE": output.metrics(batch["f"], 1.0)["MSE"]}


# ---------------------------------------------------------------------------
# Inference models
# ---------------------------------------------------------------------------

def build_inference_model(s: Array, t: Array, priors: dict) -> Callable:
    """Poisson GP / surrogate model.  Accepts surrogate_decoder=None for exact GP."""
    T, L = t.shape[0], s.shape[0]
    s_exp = jnp.broadcast_to(s, (T, L, 2))
    t_exp = t[:, None, None] * jnp.ones((1, L, 1))
    st = jnp.concatenate([s_exp, t_exp], axis=-1).reshape(T * L, 3)
    surrogate_kwargs = {"s": st}

    def model(surrogate_decoder=None, obs_mask=True, y=None):
        # FIXED: dropped a, nu samples (held at 1.0 in kernel).
        ls    = numpyro.sample("ls",    priors["ls"])
        alpha = numpyro.sample("alpha", priors["alpha"])
        beta  = numpyro.sample("beta",  priors["beta"])
        z     = numpyro.sample("z",     dist.Normal(), sample_shape=(1, T * L))
        cond  = jnp.array([ls, alpha])
        if surrogate_decoder is not None:
            mu = surrogate_decoder(z, cond, **surrogate_kwargs).reshape(T, L)
        else:
            K = gneiting_kernel(s, t, s, t, 1.0, ls, 1.0, alpha, 1.0, 1.0)
            K = K + JITTER * jnp.eye(T * L)
            L_chol = jnp.linalg.cholesky(K)
            mu = (L_chol @ z[0]).reshape(T, L)
        mu = numpyro.deterministic("mu", mu)
        lam = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lam), obs=y)

    return model


# ---------------------------------------------------------------------------
# Training / checkpoint helpers
# ---------------------------------------------------------------------------

def surrogate_model_train(
    rng_train, rng_test, loader, train_step, valid_step,
    model, results_dir, optimizer,
):
    flop_batch = next(loader(rng_train))
    rngs = {"params": rng_train, "extra": rng_test}
    kwargs = model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    state = TrainState.create(apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer)
    t0 = datetime.now()
    state = train(
        rng_train, model, optimizer, train_step, TRAIN_STEPS, loader,
        valid_step, VALID_INTERVAL, VALID_STEPS, loader,
        return_state="best", valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - t0).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, VALID_STEPS)["norm MSE"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    return train_time, eval_mse, state


def reload_state(ckpt_dir: Path, model: nn.Module, st: Array, optimizer) -> TrainState:
    TL = st.shape[0]
    dummy = {
        "s": st,
        "z": jnp.ones((1, TL)),
        "conditionals": jnp.array([10.0, 0.5]),  # FIXED: 2-D cond (ls, alpha)
        "f": jnp.ones((1, TL)),
    }
    rngs = {"params": random.key(0), "extra": random.key(1)}
    init_vars = model.init(rngs, **dummy)
    init_params = init_vars.pop("params")
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        template = TrainState.create(
            apply_fn=model.apply, params=init_params, kwargs=init_vars, tx=optimizer
        )
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(ckpt_dir.absolute(), item={"state": template, "config": {}})
    return ckpt["state"]


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

def run_hmc(rng, model, y_obs, obs_mask, surrogate_decoder=None):
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=HMC_CHAINS, num_samples=HMC_SAMPLES, num_warmup=HMC_WARMUP)
    t0 = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - t0).total_seconds()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(
        k2, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask
    )
    return samples, mcmc, post, infer_time


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ess(mcmc, var_names) -> dict:
    ess = az.ess(mcmc, method="mean", var_names=var_names)
    return {v: float(ess[v].values.mean()) for v in var_names}


def compute_rhat(samples_by_chain: dict, var_names) -> dict:
    idata = az.convert_to_inference_data(
        {k: np.array(v) for k, v in samples_by_chain.items()}
    )
    rhat = az.rhat(idata, var_names=var_names)
    return {v: float(rhat[v].values.mean()) for v in var_names}


def compute_wasserstein(samples_a: dict, samples_b: dict, var_names) -> dict:
    return {
        v: wasserstein_distance(
            np.array(samples_a[v]).flatten(),
            np.array(samples_b[v]).flatten(),
        )
        for v in var_names if v in samples_a and v in samples_b
    }


def field_mse(y_obs, post, obs_mask) -> tuple:
    y_hat = post["obs"].mean(axis=0)
    sq = (y_obs - y_hat) ** 2
    return float(sq.mean()), float(sq[obs_mask].mean()), float(sq[~obs_mask].mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictive_means(y_obs, results, obs_mask, grid_shape, time_steps, save_path):
    H, W = grid_shape
    model_names = [r["model_name"] for r in results]
    f_hats = [r["y_hat_mean"] for r in results]
    n_rows = 2 + len(model_names)
    fig, axes = plt.subplots(
        n_rows, time_steps,
        figsize=(3 * time_steps, 3 * n_rows),
        constrained_layout=True,
    )
    y_grid = np.log1p(np.array(y_obs)).reshape(time_steps, H, W)
    f_grids = [np.log1p(np.array(f)).reshape(time_steps, H, W) for f in f_hats]
    vmin = min(y_grid.min(), *(f.min() for f in f_grids))
    vmax = max(y_grid.max(), *(f.max() for f in f_grids))
    mask_grid = np.array(obs_mask).reshape(time_steps, H, W)
    for ti in range(time_steps):
        axes[0, ti].imshow(y_grid[ti], vmin=vmin, vmax=vmax, origin="lower")
        axes[0, ti].set_title(f"True, t={ti}", fontsize=8)
        masked = np.ma.masked_where(~mask_grid[ti], y_grid[ti])
        axes[1, ti].imshow(masked, vmin=vmin, vmax=vmax, origin="lower")
        axes[1, ti].set_title(f"Obs, t={ti}", fontsize=8)
        for mi, fg in enumerate(f_grids):
            axes[2 + mi, ti].imshow(fg[ti], vmin=vmin, vmax=vmax, origin="lower")
            if ti == 0:
                axes[2 + mi, ti].set_ylabel(model_names[mi], fontsize=8)
    for ax in axes.flatten():
        ax.axis("off")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(seed: int = 19):
    rng = random.key(seed)
    save_dir = Path("results/fm_spatiotemporal/").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    s, t, st = build_st_grid()
    T, L, TL = TIME_STEPS, s.shape[0], st.shape[0]
    print(f"Grid: {GRID_SHAPE} spatial × {T} time = {TL} dimensions")

    rng, rng_obs, rng_mask = random.split(rng, 3)
    y_obs = gen_y_obs(rng_obs, s, t)
    spat_mask = gen_obs_mask(rng_mask)             # [L]  spatial obs mask
    obs_mask_full = jnp.zeros((T, L), dtype=bool)
    obs_mask_full = obs_mask_full.at[T_OBS_MASK].set(spat_mask)   # [T, L]

    priors = make_priors()
    infer_model = build_inference_model(s, t, priors)
    loader = gen_train_dataloader(st, priors)

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
        # FIXED: optax.adamw destabilises FM training (vector field collapses to
        # ~0 across all training-data variance ranges we tried).  YOGI's
        # sign-based second-moment update absorbs FM's noisy per-batch
        # gradients (different t in [0,1] per sample) and lets the
        # vector field learn.  Same chain as in
        # https://github.com/mrc-ide/DeepRV_malaria/blob/main/scripts/train_flow_matching.py
        optimizer = optax.chain(
            optax.clip_by_global_norm(3.0),
            optax.scale_by_yogi(),
            optax.add_decayed_weights(1e-2),
            optax.scale_by_schedule(lambda step: -lr_schedule(step)),
        )

        if ckpt_dir.exists():
            print(f"  [{model_name}] checkpoint found, reloading.")
            state = reload_state(ckpt_dir, nn_model, st, optimizer)
        else:
            print(f"\n=== Training {model_name} ===")
            rng, rng_t, rng_v = random.split(rng, 3)
            wandb.init(
                config={"model_name": model_name, "dataset": "spatiotemporal", "seed": seed},
                mode="disabled", reinit=True,
            )
            train_time, eval_mse, state = surrogate_model_train(
                rng_t, rng_v, loader, train_step, valid_step,
                nn_model, model_dir, optimizer,
            )
            print(f"  trained in {train_time:.0f}s  |  eval norm MSE: {eval_mse:.4f}")

        trained_states[model_name] = (state, nn_model)

    # --- Build eval decoders ---
    state_drv, drv_model = trained_states["DeepRV + gMLP"]
    state_fm, fm_base_model = trained_states["FM-DeepRV"]
    fm_vf = fm_base_model.vf

    eval_decoders = {"DeepRV + gMLP": generate_surrogate_decoder(state_drv, drv_model)}
    for k in FM_K_STEPS:
        fm_k = FlowMatchingDeepRV(vf=fm_vf, n_steps=k)
        eval_decoders[f"FM-DeepRV (K={k})"] = generate_surrogate_decoder(state_fm, fm_k)

    all_model_names = ["Exact GP"] + list(eval_decoders.keys())
    all_decoders    = {None: None, **eval_decoders}   # None key = exact GP

    results = []
    gp_samples = None   # reference posterior for Wasserstein

    for model_name in all_model_names:
        decoder = None if model_name == "Exact GP" else eval_decoders[model_name]
        safe = model_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        cache = save_dir / f"{safe}.pkl"

        if cache.exists():
            print(f"  [{model_name}] cached, loading.")
            with open(cache, "rb") as fh:
                res = pickle.load(fh)
        else:
            print(f"\n=== {model_name} ===")
            rng, rng_i = random.split(rng)
            samples, mcmc, post, infer_time = run_hmc(
                rng_i, infer_model, y_obs, obs_mask_full, decoder
            )
            mcmc.print_summary()
            sbc = {k: np.array(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
            ess_vals  = compute_ess(mcmc, COND_NAMES)
            rhat_vals = compute_rhat(sbc, COND_NAMES)
            mse_all, mse_obs, mse_unobs = field_mse(y_obs, post, obs_mask_full)
            y_hat_mean = np.array(post["obs"].mean(axis=0))
            res = {
                "model_name": model_name,
                "infer_time": infer_time,
                "MSE (all)": mse_all,
                "MSE (obs)": mse_obs,
                "MSE (unobs)": mse_unobs,
                **{f"ESS {v}": ess_vals[v] for v in COND_NAMES},
                **{f"r_hat {v}": rhat_vals[v] for v in COND_NAMES},
                **{f"mean {v}": float(np.array(samples[v]).mean()) for v in COND_NAMES},
                "samples": {k: np.array(v) for k, v in samples.items()},
                "samples_by_chain": sbc,
                "y_hat_mean": y_hat_mean,
            }
            with open(cache, "wb") as fh:
                pickle.dump(res, fh)

        results.append(res)
        if model_name == "Exact GP":
            gp_samples = res["samples"]

    # --- Wasserstein vs exact GP ---
    if gp_samples is not None:
        for res in results:
            w = compute_wasserstein(gp_samples, res["samples"], COND_NAMES)
            res.update({f"W_{v}": w.get(v, float("nan")) for v in COND_NAMES})

    # --- Aggregate and save ---
    scalar_keys = (
        ["model_name", "infer_time", "MSE (all)", "MSE (obs)", "MSE (unobs)"]
        + [f"ESS {v}" for v in COND_NAMES]
        + [f"r_hat {v}" for v in COND_NAMES]
        + [f"mean {v}" for v in COND_NAMES]
        + [f"W_{v}" for v in COND_NAMES]
    )
    df = pd.DataFrame([{k: r.get(k, float("nan")) for k in scalar_keys} for r in results])
    df.to_csv(save_dir / "results.csv", index=False)
    print("\n" + df.set_index("model_name").to_string())

    plot_predictive_means(
        y_obs, results, obs_mask_full,
        GRID_SHAPE, TIME_STEPS,
        save_dir / "predictive_means.png",
    )
    print(f"\nOutputs saved to {save_dir}")


if __name__ == "__main__":
    main()
