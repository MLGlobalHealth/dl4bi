#!/usr/bin/env python3
import os
from collections.abc import Callable
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
import pandas as pd
from jax import random
from jax.scipy.stats import norm
from matplotlib.axes import Axes
from omegaconf import DictConfig, OmegaConf

import wandb
from dl4bi.meta_regression.train_utils import (
    TrainState,
    cfg_to_run_name,
    cosine_annealing_lr,
    instantiate,
    save_ckpt,
    train,
)

# NOTE: uncomment to speed up on NVIDIA GPUs
# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#code-generation-flags
# os.environ["XLA_FLAGS"] = (
# "--xla_gpu_enable_triton_softmax_fusion=true "
# "--xla_gpu_triton_gemm_any=True "
# "--xla_gpu_enable_async_collectives=true "
# "--xla_gpu_enable_latency_hiding_scheduler=true "
# "--xla_gpu_enable_highest_priority_async_stream=true "
# )


@hydra.main("configs/heaton", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    state = None
    for train_num_steps in cfg.train_num_steps:
        rng_data, rng_train, rng_test, rng = random.split(rng, 4)
        train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(
            rng_data,
            cfg.data.path,
            cfg.data.valid_pct,
            cfg.data.num_ctx.min,
            cfg.data.num_ctx.max,
            cfg.data.num_test.max,
        )
        lr_schedule = cosine_annealing_lr(
            train_num_steps,
            cfg.lr_peak,
            cfg.lr_pct_warmup,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.clip_max_norm),
            optax.yogi(lr_schedule),
        )
        model = instantiate(cfg.model)
        state = train(
            rng_train,
            model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            train_num_steps,
            cfg.valid_num_steps,
            cfg.valid_interval,
            state=state,
        )
        log_test_results(rng_test, state, test_dataloader)
    path = Path(f"results/heaton/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    path: Path,
    valid_pct: float = 0.10,
    num_ctx_min: int = 200,
    num_ctx_max: int = 500,
    num_test_max: int = 1000,
):
    """
    The image consists of ~105k observed locations and ~45k unobserved
    locations. For training, we partition the 105k observed locations into train
    and validation datasets.
    """
    df = pd.read_csv(path)
    df.Lat -= df.Lat.mean()
    df.Lon -= df.Lon.mean()
    s_obs, f_obs, s_unobs, f_unobs = split_observed(df)
    B, L, L_train = 4, s_obs.shape[0], int((1 - valid_pct) * s_obs.shape[0])
    permute_idx = random.choice(rng, L, (L,), replace=False)
    s_obs, f_obs = s_obs[permute_idx, :], f_obs[permute_idx, :]
    s_train, f_train = s_obs[:L_train, :], f_obs[:L_train, :]
    s_valid, f_valid = s_obs[L_train:, :], f_obs[L_train:, :]
    valid_lens_test = jnp.repeat(num_test_max, B)

    def train_dataloader(rng: jax.Array):
        reflections = jnp.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
        while True:
            rng_permute, rng_valid, rng = random.split(rng, 3)
            s_ctxs, f_ctxs, s_tests, f_tests = [], [], [], []
            for reflection in reflections:
                rng_i, rng_permute = random.split(rng_permute)
                permute_idx = random.choice(rng_i, L_train, (L_train,), replace=False)
                s_perm, f_perm = s_train[permute_idx, :], f_train[permute_idx, :]
                s_perm *= reflection
                s_ctxs += [s_perm[:num_ctx_max, :]]
                f_ctxs += [f_perm[:num_ctx_max, :]]
                # NOTE: (s,f)_test are superset of (s,f)_ctx
                s_tests += [s_perm[:num_test_max, :]]
                f_tests += [f_perm[:num_test_max, :]]
            valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
            yield (
                jnp.stack(s_ctxs),
                jnp.stack(f_ctxs),
                valid_lens_ctx,
                jnp.stack(s_tests),
                jnp.stack(f_tests),
                valid_lens_test,
            )

    def valid_dataloader(rng: jax.Array):
        yield (
            s_train[None, ...],  # add dummy batch dim
            f_train[None, ...],
            jnp.array([L_train]),  # use all train locations
            s_valid[None, ...],
            f_valid[None, ...],
            jnp.array([s_valid.shape[0]]),  # all test points are valid
        )

    def test_dataloader(rng: jax.Array):
        yield (
            s_obs[None, ...],  # add a dummy batch dim
            f_obs[None, ...],
            jnp.array([L]),  # use all observed locations
            s_unobs[None, ...],
            f_unobs[None, ...],
            jnp.array([s_unobs.shape[0]]),  # all test points are valid
        )

    return train_dataloader, valid_dataloader, test_dataloader


def split_observed(df: pd.DataFrame):
    """Splits `col` into observed and unobserved locations."""
    obs_idx = df.MaskTemp.notna().values
    obs = df[obs_idx][["Lon", "Lat", "MaskTemp"]].values
    unobs = df[~obs_idx][["Lon", "Lat", "TrueTemp"]].values
    s_obs, f_obs = obs[:, :-1], obs[:, [-1]]
    s_unobs, f_unobs = unobs[:, :-1], unobs[:, [-1]]
    return s_obs, f_obs, s_unobs, f_unobs


def log_test_results(rng: jax.Array, state: TrainState, test_dataloader: Callable):
    """Logs a plot of the entire image to wandb."""
    rng_data, rng = random.split(rng)
    batch = next(test_dataloader(rng_data))
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test = batch
    rng_dropout, rng_extra = random.split(rng)
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    # remove dummy batch dimension
    s_ctx, f_ctx, s_test, f_test = s_ctx[0], f_ctx[0], s_test[0], f_test[0]
    f_mu, f_std = f_mu[0], f_std[0]
    log_metrics(f_test, f_mu, f_std)
    s = jnp.vstack([s_ctx, s_test])
    f_task = jnp.vstack([f_ctx, jnp.full(f_mu.shape, jnp.nan)])
    f_pred = jnp.vstack([f_ctx, f_mu])
    f_true = jnp.vstack([f_ctx, f_test])
    data = jnp.hstack([s, f_task, f_pred, f_true])
    df = pd.DataFrame(data, columns=["Lon", "Lat", "Task", "Pred", "True"])
    log_plot(df)


# TODO(danj): implement "interval score", and CRPS
def log_metrics(
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    hdi_prob: float = 0.95,
):
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    nll = -norm.logpdf(f_test, f_mu, f_std).mean()
    rmse = jnp.sqrt(jnp.square(f_test - f_mu).mean())
    mae = jnp.abs(f_test - f_mu).mean()
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    cvg = ((f_test >= f_lower) & (f_test <= f_upper)).mean()
    wandb.log(
        {
            "Test NLL": nll,
            "Test RSME": rmse,
            "Test MAE": mae,
            "Test Coverage": cvg,
        }
    )


def log_plot(df: pd.DataFrame):
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot(df, "Task", axs[0])
    plot(df, "Pred", axs[1])
    plot(df, "True", axs[2])
    path = "/tmp/heaton_benchmark.png"
    plt.tight_layout()
    plt.savefig(path, dpi=125)
    plt.clf()
    wandb.log({"Heaton Benchmark": wandb.Image(path)})


def plot(df: pd.DataFrame, col: str, ax: Axes | None = None):
    """Plots a satellite image from a pandas DataFrame."""
    if not isinstance(ax, Axes):
        ax = plt.gca()
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    ax.imshow(df[col].values.reshape(300, 500), cmap=cmap, interpolation="none")
    ax.set_title(col)
    return plt.gcf()


if __name__ == "__main__":
    main()
