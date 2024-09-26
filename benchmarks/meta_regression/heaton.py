#!/usr/bin/env python3
import os
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
    Callback,
    TrainState,
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    mask_from_valid_lens,
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
    rng_train, rng_data, rng_test = random.split(rng, 3)
    train_dataloader, valid_dataloader = build_dataloaders(
        rng_data,
        cfg.data.path,
        cfg.data.valid_pct,
        cfg.data.batch_size,
        cfg.data.num_ctx.min,
        cfg.data.num_ctx.max,
        cfg.data.num_test.max,
    )
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
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
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[Callback(log_plot, cfg.plot_interval)],
    )
    metrics = evaluate(rng_test, state, valid_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/heaton/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    path: Path,
    valid_pct: float = 0.10,
    batch_size: int = 16,
    num_ctx_min: int = 200,
    num_ctx_max: int = 500,
    num_test_max: int = 1000,
):
    """
    The image consists of ~105k observed locations and ~45k unobserved locations.
    Here we partition the 105k observed locations into train and validation datasets.
    With each batch, we also pass all observed locations and unobserved locations
    so that the plotting callback can use them periodically.
    """
    df = pd.read_csv(path)
    df = preprocess(df)
    s_obs, f_obs, s_unobs = split_observed(df)
    B, L, L_train = batch_size, s_obs.shape[0], int((1 - valid_pct) * s_obs.shape[0])
    permute_idx = random.choice(rng, L, (L,), replace=False)
    s_obs, f_obs = s_obs[permute_idx, :], f_obs[permute_idx, :]
    s_train, f_train = s_obs[:L_train, :], f_obs[:L_train, :]
    s_valid, f_valid = s_obs[L_train:, :], f_obs[L_train:, :]
    valid_lens_test = jnp.repeat(num_test_max, B)

    def train_dataloader(rng: jax.Array):
        while True:
            rng_permute, rng_valid, rng = random.split(rng, 3)
            s_ctxs, f_ctxs, s_tests, f_tests = [], [], [], []
            for _ in range(B):  # TODO(danj): speed up?
                rng_i, rng_permute = random.split(rng_permute)
                permute_idx = random.choice(rng_i, L_train, (L_train,), replace=False)
                s_perm, f_perm = s_train[permute_idx, :], f_train[permute_idx, :]
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
                s_obs,  # return full originals for log_plot callback
                f_obs,
                s_unobs,
            )

    def valid_dataloader(rng: jax.Array):
        yield (
            s_train[None, ...],  # add dummy batch dim
            f_train[None, ...],
            jnp.array([L_train]),  # use all train locations
            s_valid[None, ...],
            f_valid[None, ...],
            jnp.array([s_valid.shape[0]]),  # use all valid locations
        )

    return train_dataloader, valid_dataloader


def preprocess(df: pd.DataFrame):
    """De-mean locations and standardize temperature."""
    df.Lon -= df.Lon.mean()
    df.Lat -= df.Lat.mean()
    df.Temp = (df.Temp - df.Temp.mean()) / df.Temp.std()
    return df


def split_observed(df: pd.DataFrame):
    """Splits the df into observed and unobserved locations."""
    obs_idx = df.Temp.notna().values
    obs, unobs = df[obs_idx].values, df[~obs_idx].values
    s_obs, f_obs = obs[:, :-1], obs[:, [-1]]
    s_unobs = unobs[:, :-1]  # f_unobs is all nans
    return s_obs, f_obs, s_unobs


def log_plot(step: int, rng_step: jax.Array, state: TrainState, batch: tuple):
    """Logs a plot of the entire image to wandb."""
    *_, s_ctx, f_ctx, s_test = batch
    rng_dropout, rng_extra = random.split(rng_step)
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx[None, ...],  # add dummy batch dimension
        f_ctx[None, ...],
        s_test[None, ...],
        valid_lens_ctx=None,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    f_mu = f_mu[0, ...]  # remove dummy batch dimension
    s = jnp.vstack([s_ctx, s_test])
    f_pred = jnp.vstack([f_ctx, f_mu])
    f_task = jnp.vstack([f_ctx, jnp.full(f_mu.shape, jnp.nan)])
    data = jnp.hstack([s, f_task, f_pred])
    df = pd.DataFrame(data, columns=["Lon", "Lat", "Task", "Pred"])
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot(df, "Task", axs[0])
    plot(df, "Pred", axs[1])
    # TODO(danj): add plot(df, 'GroundTruth', axs[2])
    path = f"/tmp/heaton_step_{step}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=125)
    plt.clf()
    wandb.log({f"Step {step}": wandb.Image(path)})


def plot(df: pd.DataFrame, col="Temp", ax: Axes | None = None):
    """Plots a satellite image from a pandas DataFrame."""
    if not isinstance(ax, Axes):
        ax = plt.gca()
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    ax.imshow(df[col].values.reshape(300, 500), cmap=cmap, interpolation="none")
    ax.set_title(col)
    return plt.gcf()


# TODO(danj): implement "interval score", and CRPS
def log_metrics(step: int, rng_step: jax.Array, state: TrainState, batch: tuple):
    s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *_ = batch
    rng_dropout, rng_extra = random.split(rng_step)
    f_mu, f_std, *_ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx,
        f_ctx,
        s_test,
        valid_lens_ctx,
        valid_lens_test=None,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    hdi_prob = 0.95
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower, f_upper = f_mu - z_score * f_std, f_mu + z_score * f_std
    # by construction (see dataloader above), s_test contains all
    # context points and then some extra non-context test locations;
    # here we want to ignore the context points in the test set and
    # calculate metrics for non-context test locations.
    mask_test = ~mask_from_valid_lens(s_test.shape[1], valid_lens_ctx)
    rmse = jnp.sqrt(jnp.square(f_test - f_mu).mean(where=mask_test))
    mae = jnp.abs(f_test - f_mu).mean(where=mask_test)
    cvg = ((f_test >= f_lower) & (f_test <= f_upper)).mean(where=mask_test)
    wandb.log({"RMSE": rmse, "MAE": mae, "Coverage": cvg})


if __name__ == "__main__":
    main()
