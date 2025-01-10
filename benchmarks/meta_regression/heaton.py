#!/usr/bin/env python3
import math
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax
import pandas as pd
import wandb
from jax import random, vmap
from jax.scipy.stats import norm
from matplotlib.axes import Axes
from omegaconf import DictConfig, OmegaConf
from optax.schedules import cosine_decay_schedule
from sps.utils import random_subgrid
from tqdm import tqdm

from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cfg_to_run_name,
    evaluate,
    instantiate,
    load_ckpt,
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
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    if not Path(cfg.data.train_rngs_path).exists():
        generate_train_rngs(rng_train, cfg.train_num_steps, cfg.kernel, cfg.data)
    dataloaders = build_dataloaders(cfg.data, cfg.kernel, cfg.test)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders
    num_decay_steps = int(cfg.lr_pct_decay * cfg.train_num_steps)
    lr_schedule = cosine_decay_schedule(cfg.lr_peak, num_decay_steps, cfg.lr_alpha)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    H, W = cfg.data.s[0].num, cfg.data.s[1].num
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    callback = Callback(
        partial(log_img_plots, shape=(H, W, 1), cmap=cmap),
        cfg.plot_interval,
    )
    state = None
    finetune_path = cfg.get("finetune", None)
    train_num_steps = cfg.train_num_steps
    if finetune_path:
        state, _ = load_ckpt(Path(finetune_path))
        optimizer = optax.yogi(cfg.lr_finetune)
        train_num_steps = cfg.finetune_num_steps
        if cfg.finetune_on_real:
            train_dataloader = valid_dataloader
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[callback],
        monitor_metric=cfg.monitor_metric,
        early_stop_patience=cfg.early_stop_patience,
        state=state,
    )
    metrics = evaluate(rng_test, state, valid_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))
    # NOTE: uncomment to run actual test
    log_test_results(rng_test, state, test_dataloader)


def build_dataloaders(data: DictConfig, kernel: DictConfig, test: DictConfig):
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
    del df  # save memory
    L_obs, L_unobs = s_obs.shape[0], s_unobs.shape[0]
    L_train = jnp.prod(jnp.array([dim.num for dim in data.s]))
    num_ctx_min = math.floor((1 - data.max_masked_pct) * L_train)
    num_ctx_max = math.ceil((1 - data.min_masked_pct) * L_train)
    # NOTE: these reflections assume the data is centered on the origin (0,)*D
    reflections = jnp.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])[:, None, :]
    B, N_r = data.batch_size, len(reflections)
    mB = B // N_r

    def build_train_dataloader():
        """Generates batches of random subgrids."""
        valid_lens_test = jnp.repeat(L_train, B)
        var = kernel.kwargs.var.kwargs.kwargs.mu
        mask_threshold = norm.ppf(1 - data.mask_pct, loc=0, scale=var)

        def gen_batch(rng: jax.Array):
            rng, s, f = sample_constrained_s_f(rng, kernel, data)
            rng_eps, _ = random.split(rng)
            # use the next image in the batch to mask the previous
            rot_idx = jnp.arange(1, mB + 1).at[-1].set(0)
            f_mask = f[rot_idx] > mask_threshold
            valid_lens_ctx = L_train - jnp.repeat(f_mask.sum(axis=(1, 2)), N_r)
            f_masked = f.at[f_mask].set(jnp.nan)
            sort_idx = vmap(jnp.argsort)(f_masked.squeeze())  # nans to end
            inv_sort_idx = vmap(jnp.argsort)(sort_idx)
            # use reflections to convert minibatch to full batch
            ss, fs = [], []
            for i in range(mB):
                ss += [jnp.stack([s[sort_idx[i], :]] * N_r) * reflections]
                fs += [jnp.stack([f[i, sort_idx[i], :]] * N_r)]
            s_ord, f_ord = jnp.vstack(ss), jnp.vstack(fs)
            f_ord_noisy = f_ord + data.obs_noise * random.normal(rng_eps, f_ord.shape)
            return (
                s_ord[:, :num_ctx_max, :],
                f_ord_noisy[:, :num_ctx_max, :],
                valid_lens_ctx,
                s_ord,
                f_ord,
                valid_lens_test,
                # also pass full, unmodified originals for plotting
                jnp.vstack([jnp.stack([s] * N_r) * reflections] * mB),
                jnp.repeat(f, N_r, axis=0),
                jnp.repeat(inv_sort_idx, N_r, axis=0),
            )

        def dataloader(rng: jax.Array):
            while True:
                rng_batch, rng = random.split(rng)
                yield gen_batch(rng_batch)

        def build_deterministic_dataloader(rngs: jax.Array):
            def dataloader(_rng: jax.Array):
                for rng in rngs:
                    yield gen_batch(rng)

            return dataloader

        loader = dataloader
        train_rngs_path = data.get("train_rngs_path")
        if Path(train_rngs_path).exists():
            train_rngs = jnp.load(train_rngs_path, allow_pickle=True)
            loader = build_deterministic_dataloader(train_rngs)

        return loader

    def valid_dataloader(rng: jax.Array):
        valid_lens_test = jnp.repeat(L_train, B)
        while True:
            rng_idx, rng_valid, rng = random.split(rng, 3)
            ss, fs = [], []
            for i in range(mB):
                rng_i, rng_idx = random.split(rng_idx)
                idx = random.choice(rng_i, L_obs, (L_train,), replace=False)
                s, f = s_obs[idx], f_obs[idx]
                s = jnp.stack([s] * N_r) * reflections
                f = jnp.stack([f] * N_r)
                ss += [s]
                fs += [f]
            s, f = jnp.vstack(ss), jnp.vstack(fs)
            valid_lens_ctx = random.randint(rng_valid, (B,), num_ctx_min, num_ctx_max)
            yield (
                s[:, :num_ctx_max, :],
                f[:, :num_ctx_max, :],
                valid_lens_ctx,
                s,
                f,
                valid_lens_test,
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
    return (
        jnp.float16(s_obs),
        jnp.float16(f_obs),
        jnp.float16(s_unobs),
        jnp.float16(f_unobs),
    )


def generate_train_rngs(
    rng: jax.Array,
    num_batches: int,
    kernel: DictConfig,
    data: DictConfig,
):
    rngs = []
    pbar = tqdm(range(1, num_batches + 1), unit=" batches", dynamic_ncols=True)
    for i in pbar:
        rng_i, rng = random.split(rng)
        rng, *_ = sample_constrained_s_f(rng, kernel, data)
        rngs += [rng]
    jnp.save(data.train_rngs_path, jax.random.key_data(jnp.stack(rngs)))


def sample_constrained_s_f(rng: jax.Array, kernel: DictConfig, data: DictConfig):
    valid_pct, var = 1 - data.mask_pct, kernel.kwargs.var.kwargs.kwargs.mu
    threshold = norm.ppf(valid_pct, loc=0, scale=var)
    s, f = sample_s_f(rng, kernel, data)
    pct_masked = (f > threshold).mean(axis=(1, 2))
    while jnp.logical_or(
        pct_masked < data.min_masked_pct,
        pct_masked > data.max_masked_pct,
    ).any():
        rng, _ = random.split(rng)
        s, f = sample_s_f(rng, kernel, data)
        pct_masked = (f > threshold).mean(axis=(1, 2))
    return rng, s, f


def sample_s_f(rng: jax.Array, kernel: DictConfig, data: DictConfig):
    rng_s, rng_f = random.split(rng)
    s = random_subgrid(rng_s, data.s, data.min_axes_pct, data.max_axes_pct)
    s = s.reshape(-1, s.shape[-1])
    gp = instantiate(kernel)
    # divide by 4 to account for reflections added later
    f, *_ = gp.simulate(rng_f, s, data.batch_size // 4)
    return s, f


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
            "Heaton Test NLL": nll,
            "Heaton Test RSME": rmse,
            "Heaton Test MAE": mae,
            "Heaton Test Coverage": cvg,
        }
    )


def log_plot(df: pd.DataFrame, wandb_key: str = "Heaton Benchmark"):
    df = df.sort_values(["Lat", "Lon"], ascending=[False, True])
    shape = (df.Lat.nunique(), df.Lon.nunique())
    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot(df, "Task", axs[0], shape)
    plot(df, "Pred", axs[1], shape)
    plot(df, "True", axs[2], shape)
    path = f"/tmp/{datetime.now().isoformat()} heaton_benchmark.png"
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
