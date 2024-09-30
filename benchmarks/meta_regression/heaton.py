#!/usr/bin/env python3
from collections.abc import Callable
from functools import partial
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
from sps.utils import random_subgrid

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cfg_to_run_name,
    cosine_annealing_lr,
    instantiate,
    log_img_plots,
    save_ckpt,
    train,
)


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
    rng_data, rng_train, rng_finetune, rng_test, rng = random.split(rng, 5)
    dataloaders = build_dataloaders(rng_data, cfg.data, cfg.kernel, cfg.test)
    train_dataloader, finetune_dataloader, valid_dataloader, test_dataloader = (
        dataloaders
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
    H, W = cfg.data.s[0].num, cfg.data.s[1].num
    callback = Callback(partial(log_img_plots, shape=(H, W, 1)), cfg.plot_interval)
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[callback],
    )
    state = train(
        rng_finetune,
        model,
        optax.yogi(1e-5),
        finetune_dataloader,
        valid_dataloader,
        cfg.finetune_num_steps,
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
    data: DictConfig,
    kernel: DictConfig,
    test: DictConfig,
):
    """
    The image consists of ~105k observed locations and ~45k unobserved
    locations. For training, we partition the 105k observed locations into train
    and validation datasets.
    """
    df = pd.read_csv(test.path)
    df.Lat -= df.Lat.mean()
    df.Lon -= df.Lon.mean()
    mean, std = df.MaskTemp.mean(), df.MaskTemp.std()
    df["MaskTemp"] = (df.MaskTemp - mean) / std
    df["TrueTemp"] = (df.TrueTemp - mean) / std
    s_obs, f_obs, s_unobs, f_unobs = split_observed(df)
    L_obs, L_unobs = s_obs.shape[0], s_unobs.shape[0]
    L_valid = int(test.valid_pct * L_obs)
    L_finetune = L_obs - L_valid
    permute_idx = random.choice(rng, L_obs, (L_obs,), replace=False)
    s_obs, f_obs = s_obs[permute_idx], f_obs[permute_idx]
    s_valid, f_valid = s_obs[:L_valid], f_obs[:L_valid]
    s_finetune, f_finetune = s_obs[L_valid:], f_obs[L_valid:]
    L = jnp.prod(jnp.array([dim.num for dim in data.s]))
    B, D = data.batch_size, len(data.s)
    # NOTE: these reflections assume the data is centered on the origin (0,)*D
    reflections = jnp.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    N_r = len(reflections)
    mB = B // N_r

    def build_train_dataloader():
        """Generates batches of random subgrids."""
        gp = instantiate(kernel)
        valid_lens_test = jnp.repeat(L, B)  # all positions in test set

        def gen_batch(rng: jax.Array):
            rng_s, rng_f, rng_valid, rng_permute, rng_eps = random.split(rng, 5)
            s = random_subgrid(rng_s, data.s, data.min_axes_pct).reshape(-1, D)
            f, *_ = gp.simulate(rng_f, s, mB)
            f = jnp.repeat(f, N_r, axis=0)  # [B, L, D]
            s = jnp.stack([s] * N_r) * reflections[:, None, :]  # [N_r, L, D]
            s = jnp.vstack([s] * mB)  # [B, L, D]
            permute_idx = random.choice(rng_permute, L, (L,), replace=False)
            inv_permute_idx = jnp.argsort(permute_idx)
            s_perm = s[:, permute_idx, :]
            f_perm = f[:, permute_idx, :]
            valid_lens_ctx = random.randint(
                rng_valid, (B,), data.num_ctx.min, data.num_ctx.max
            )
            eps = random.normal(rng_eps, f_perm.shape)
            f_perm_noisy = f_perm + data.obs_noise * eps
            return (
                s_perm,
                f_perm_noisy,
                valid_lens_ctx,
                s_perm,
                f_perm,
                valid_lens_test,
                s,
                f,
                inv_permute_idx,
            )

        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng = random.split(rng)
                yield gen_batch(rng_batch)

        return dataloader

    def finetune_dataloader(rng: jax.Array):
        valid_lens_test = jnp.repeat(L, B)
        while True:
            rng_idx, rng_valid, rng = random.split(rng, 3)
            ss, fs = [], []
            for i in range(mB):
                rng_i, rng_idx = random.split(rng_idx)
                idx = random.choice(rng_i, L_obs, (L,), replace=False)
                s, f = s_finetune[idx], f_finetune[idx]
                s = jnp.stack([s] * N_r) * reflections[:, None, :]
                f = jnp.stack([f] * N_r)
                ss += [s]
                fs += [f]
            s, f = jnp.vstack(ss), jnp.vstack(fs)
            valid_lens_ctx = random.randint(
                rng_valid, (B,), data.num_ctx.min, data.num_ctx.max
            )
            yield s, f, valid_lens_ctx, s, f, valid_lens_test

    def valid_dataloader(rng: jax.Array):
        yield (
            s_finetune[None, ...],
            f_finetune[None, ...],
            jnp.array([L_finetune]),
            s_valid[None, ...],
            f_valid[None, ...],
            jnp.array([L_valid]),
        )

    def test_dataloader(rng: jax.Array):
        yield (
            s_obs[None, ...],
            f_obs[None, ...],
            jnp.array([L_obs]),
            s_unobs[None, ...],
            f_unobs[None, ...],
            jnp.array([L_unobs]),
        )

    return (
        build_train_dataloader(),
        finetune_dataloader,
        valid_dataloader,
        test_dataloader,
    )


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
    # remove batch dimension
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


def log_plot(df: pd.DataFrame, wandb_key: str = "Heaton Benchmark"):
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    shape = (df.Lat.nunique(), df.Lon.nunique())
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot(df, "Task", axs[0], shape)
    plot(df, "Pred", axs[1], shape)
    plot(df, "True", axs[2], shape)
    path = "/tmp/heaton_benchmark.png"
    plt.tight_layout()
    plt.savefig(path, dpi=125)
    plt.clf()
    wandb.log({wandb_key: wandb.Image(path)})


def plot(
    df: pd.DataFrame,
    col: str,
    ax: Axes | None = None,
    shape: tuple[int, int] = (300, 500),
):
    """Plots a satellite image from a pandas DataFrame."""
    if not isinstance(ax, Axes):
        ax = plt.gca()
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    ax.imshow(df[col].values.reshape(shape), cmap=cmap, interpolation="none")
    ax.set_title(col)
    return plt.gcf()


if __name__ == "__main__":
    main()
