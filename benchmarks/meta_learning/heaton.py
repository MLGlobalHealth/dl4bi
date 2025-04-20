#!/usr/bin/env python3
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
from hydra.utils import instantiate
from jax import random
from jax.scipy.stats import norm
from matplotlib.axes import Axes
from omegaconf import DictConfig, OmegaConf
from sps.utils import random_subgrid

from dl4bi.core.train import (
    Callback,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatial import SpatialBatch, SpatialData
from dl4bi.meta_learning.utils import (
    cfg_to_run_name,
    wandb_2d_img_callback,
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
    dataloaders = build_dataloaders(cfg.data, cfg.kernel, cfg.test)
    train_dataloader, valid_dataloader, test_dataloader = dataloaders
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    clbk = Callback(
        partial(wandb_2d_img_callback, cmap=cmap),
        cfg.plot_interval,
    )
    state = None
    finetune_path = cfg.get("finetune_path", None)
    train_num_steps = cfg.train_num_steps
    if finetune_path:
        state, _ = load_ckpt(Path(finetune_path))
        train_num_steps = cfg.finetune_num_steps
        if cfg.finetune_on_real:
            train_dataloader = valid_dataloader
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
        callbacks=[clbk],
        callback_dataloader=train_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        num_steps=1,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(data: DictConfig, kernel: DictConfig, test: DictConfig):
    """
    The image consists of ~105k observed locations and ~45k unobserved
    locations. For training, we partition the 105k observed locations into train
    and validation datasets.
    """
    s_obs, f_obs, s_unobs, f_unobs = load_data(test.path)
    B, L_obs, L_unobs = data.batch_size, s_obs.shape[0], s_unobs.shape[0]
    L_train = jnp.prod(jnp.array([dim.num for dim in data.s]))
    H, W = data.s[0].num, data.s[1].num

    def train_dataloader(rng: jax.Array):
        """Generates batches of random subgrids."""
        inv_permute_idx = jnp.arange(L_train)
        gp = instantiate(kernel)
        while True:
            rng_s, rng_f, rng = random.split(rng, 3)
            s = random_subgrid(rng_s, data.s, data.min_axes_pct, data.max_axes_pct)
            s = s.reshape(-1, s.shape[-1])  # [L, D_s]
            f, *_ = gp.simulate(rng_f, s, B)  # [B, L, D_f]
            # use the next image in the batch to mask the previous
            rot_idx = jnp.arange(1, B + 1).at[-1].set(0)
            s = jnp.repeat(s[None, ...], B, axis=0)
            threshold = jnp.quantile(f[rot_idx], data.ctx_pct, axis=1, keepdims=True)
            mask_ctx = (f[rot_idx] < threshold)[..., 0]  # [B, K]
            yield SpatialBatch(
                x_ctx=None,
                s_ctx=s,
                f_ctx=f,
                mask_ctx=mask_ctx,
                x_test=None,
                s_test=s,
                f_test=f,
                mask_test=~mask_ctx,
                inv_permute_idx=inv_permute_idx,
                s_shape=(B, H, W, 2),
            )

    def valid_dataloader(rng: jax.Array):
        num_ctx_min, num_ctx_max = int(L_obs * 0.05), int(L_obs * 0.25)
        num_test = int(L_obs * 0.25)
        d = SpatialData(x=None, s=s_obs[None, ...], f=f_obs[None, ...])
        while True:
            rng_i, rng = random.split(rng)
            yield d.batch(
                rng_i,
                num_ctx_min,
                num_ctx_max,
                num_test,
                test_includes_ctx=False,
                batch_size=1,
            )

    def test_dataloader(rng: jax.Array):
        yield SpatialBatch(
            x_ctx=None,
            s_ctx=s_obs[None, ...],
            f_ctx=f_obs[None, ...],
            mask_ctx=jnp.ones((1, L_obs), dtype=bool),
            x_test=None,
            s_test=s_unobs[None, ...],
            f_test=f_unobs[None, ...],
            mask_test=None,
            # TODO(danj): update for plotting
            inv_permute_idx=jnp.arange(L_obs + L_unobs),
            s_shape=(),
        )

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )


def load_data(path):
    df = pd.read_csv(path)
    df.Lat -= df.Lat.mean()
    df.Lon -= df.Lon.mean()
    mean, std = df.MaskTemp.mean(), df.MaskTemp.std()
    df["MaskTemp"] = (df.MaskTemp - mean) / std
    df["TrueTemp"] = (df.TrueTemp - mean) / std
    return split_observed(df)


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


def sample_constrained_s_f(rng: jax.Array, kernel: DictConfig, data: DictConfig):
    threshold = norm.ppf(1 - data.mask_pct)
    s, f = sample_s_f(rng, kernel, data)
    pct_masked = (f > threshold).mean(axis=(1, 2))
    while jnp.logical_or(
        pct_masked < data.min_masked_pct,
        pct_masked > data.max_masked_pct,
    ).any():
        rng, _ = random.split(rng)
        s, f = sample_s_f(rng, kernel, data)
        pct_masked = (f > threshold).mean(axis=(1, 2))
    return s, f


def sample_s_f(rng: jax.Array, kernel: DictConfig, data: DictConfig):
    rng_s, rng_f = random.split(rng)
    s = random_subgrid(rng_s, data.s, data.min_axes_pct, data.max_axes_pct)
    s = s.reshape(-1, s.shape[-1])
    gp = instantiate(kernel)
    f, *_ = gp.simulate(rng_f, s, data.batch_size)
    return s, f


# TODO(danj): udpate
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
