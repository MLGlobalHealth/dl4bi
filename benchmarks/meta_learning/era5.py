#!/usr/bin/env python3
import os
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import cdsapi
import hydra
import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import xarray as xr
from hydra.utils import instantiate
from jax import random
from matplotlib.colors import Normalize
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import (
    Callback,
    TrainState,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import (
    SpatiotemporalBatch,
    SpatiotemporalData,
)
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/era5", config_name="default", version_base=None)
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
    df_train, df_valid, df_test, revert = load_data(**cfg.data.splits)
    train_dataloader = partial(dataloader, df=df_train, **cfg.data.train_dataloader)
    valid_dataloader = partial(dataloader, df=df_valid, **cfg.data.valid_dataloader)
    test_dataloader = partial(dataloader, df=df_test, **cfg.data.test_dataloader)
    callback_dataloader = partial(
        dataloader,
        df=df_valid,
        is_callback=True,
        revert=revert,
        **cfg.data.valid_dataloader,
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    clbk = Callback(plot, cfg.plot_interval)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.data.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.data.valid_interval,
        cfg.data.valid_num_steps,
        valid_dataloader,
        callbacks=[clbk],
        callback_dataloader=callback_dataloader,
        return_state="best",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        cfg.data.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def dataloader(
    rng: jax.Array,
    df: pd.DataFrame,
    batch_size: int = 16,
    num_ctx_min_per_t: int = 45,
    num_ctx_max_per_t: int = 225,
    num_test: int = 900,
    H_deg: float = 7.5,
    W_deg: float = 7.5,
    T_hrs: int = 30,
    T_hrs_delta: int = 6,
    num_batches_per_subset: int = 50,
    is_callback: bool = False,
    revert: Optional[Callable] = None,
):
    grid_res = 0.25
    H, W = int(H_deg / grid_res), int(W_deg / grid_res)
    x_cols = ["elev_std", "hour_of_day_norm"]
    s_cols = ["lat_std", "lng_std"]
    t_cols = ["hour_std"]
    f_cols = ["temp_std"]
    data_cols = x_cols + s_cols + t_cols + f_cols
    lat_uniq, lng_uniq = df.latitude.unique(), df.longitude.unique()
    lat_choices = lat_uniq[lat_uniq <= lat_uniq.max() - H_deg]
    lng_choices = lng_uniq[lng_uniq <= lng_uniq.max() - W_deg]
    while True:
        # 1. Select random start time and filter times to every T_hrs_delta
        rng_t, rng_lat, rng_lng, rng_b, rng = random.split(rng, 5)
        hr_start = random.choice(rng_t, T_hrs_delta, (1,)).item()
        dft = df[(df.valid_time.dt.hour + hr_start) % T_hrs_delta == 0]
        # 2. Select a random lat/lng region that is H_deg x W_deg
        lat_start = random.choice(rng_lat, lat_choices, (1,)).item()
        lng_start = random.choice(rng_lng, lng_choices, (1,)).item()
        dft = dft[(dft.latitude >= lat_start) & (dft.latitude < lat_start + H_deg)]
        dft = dft[(dft.longitude >= lng_start) & (dft.longitude < lng_start + W_deg)]
        # 3. Reshape data to [T, H, W, D]
        shape = (dft.valid_time.nunique(), H, W, -1)
        values = dft[data_cols].values.reshape(shape)
        # 4. Separate out x, s, t, and f
        subset = SpatiotemporalData(
            x=values[..., :2],  # [T, H, W, 2]
            s=values[..., 2:4],  # [T, H, W, 2]
            t=values[:, 0, 0, 4],  # [T]
            f=values[..., [5]],  # [T, H, W, 1]
        )
        # 5. Create a number of batches from this filtered subset
        for _ in range(num_batches_per_subset):
            rng_b, rng = random.split(rng)
            batch = subset.batch(
                rng=rng_b,
                num_t=T_hrs // T_hrs_delta,
                random_t=False,
                num_ctx_min_per_t=num_ctx_min_per_t,
                num_ctx_max_per_t=num_ctx_max_per_t,
                independent_t_masks=True,
                num_test=num_test,
                forecast=True,
                batch_size=batch_size,
            )
            yield (batch, revert) if is_callback else batch


def load_data(
    train_region: str = "central_europe",
    valid_region: str = "northern_europe",
    test_region: str = "western_europe",
):
    df_train = load_cached(train_region)
    df_valid = load_cached(valid_region)
    df_test = load_cached(test_region)
    return standardize_using_train(df_train, df_valid, df_test)


def load_cached(region: str = "central_europe"):
    download_if_not_cached()
    ds = xr.open_mfdataset(f"cache/era5/{region}/2019_*.nc", combine="by_coords")
    df = ds.to_dataframe()[["t2m", "z"]].reset_index()
    return df


def download_if_not_cached():
    """Downloads and caches netcdf files by month.

    .. note::
        You will need a CDS API key setup for this to work:
        https://cds.climate.copernicus.eu/how-to-api
    """
    client = None
    variables = ["2m_temperature", "geopotential"]
    times = [f"{t:02d}:00" for t in range(24)]
    central_europe = [53, 8, 42, 28]  # N, W, S, E
    western_europe = [53, -4, 42, 8]
    northern_europe = [62, 8, 53, 28]
    grid = [0.25, 0.25]

    for region, area in [
        ("central_europe", central_europe),
        ("western_europe", western_europe),
        ("northern_europe", northern_europe),
    ]:
        os.makedirs(f"cache/era5/{region}", exist_ok=True)
        for month in [f"{m:02d}" for m in range(1, 13)]:
            target = Path(f"cache/era5/{region}/2019_{month}.nc")
            if not target.exists():
                if client is None:
                    client = cdsapi.Client()
                print(f"\n{target} does not exist; downloading...\n")
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "format": "netcdf",
                        "variable": variables,
                        "year": "2019",
                        "month": month,
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": times,
                        "area": area,
                        "grid": grid,
                    },
                    target,
                )


def standardize_using_train(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
):
    t_min = df_train.valid_time.min()
    df_train["hour"] = (df_train.valid_time - t_min) / pd.Timedelta(hours=1)
    hour_mu, hour_std = df_train.hour.mean(), df_train.hour.std()
    lat_mu, lat_std = df_train.latitude.mean(), df_train.latitude.std()
    lng_mu, lng_std = df_train.longitude.mean(), df_train.longitude.std()
    elev_mu, elev_std = df_train.z.mean(), df_train.z.std()
    temp_mu, temp_std = df_train.t2m.mean(), df_train.t2m.std()

    def standardize(df: pd.DataFrame):
        df["hour_of_day_norm"] = df.valid_time.dt.hour / 23.0
        df["hour"] = (df.valid_time - t_min) / pd.Timedelta(hours=1)
        df["hour_std"] = (df.hour - hour_mu) / hour_std
        df["lat_std"] = (df.latitude - lat_mu) / lat_std
        df["lng_std"] = (df.longitude - lng_mu) / lng_std
        df["elev_std"] = (df.z - elev_mu) / elev_std
        df["temp_std"] = (df.t2m - temp_mu) / temp_std
        df.rename(columns={"t2m": "temp"}, inplace=True)
        # sort values this way so they can be reshaped into images of shape (T, H, W)
        df = df.sort_values(
            by=["hour_std", "lat_std", "lng_std"], ascending=[True, False, True]
        )
        return df.drop(columns=["z", "hour"]).reset_index(drop=True)

    def revert_t(t: jax.Array):
        hours = np.rint(t * hour_std + hour_mu).astype(int).astype("timedelta64[h]")
        return t_min + hours

    def revert_s(s: jax.Array):  # s: [T, H * W, 2]
        std = jnp.array([lat_std, lng_std]).reshape(1, 1, 2)
        mu = jnp.array([lat_mu, lng_mu]).reshape(1, 1, 2)
        return jnp.round(s * std + mu, decimals=1)

    return (
        standardize(df_train),
        standardize(df_valid),
        standardize(df_test),
        {"t": revert_t, "s": revert_s},
    )


def plot(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatiotemporalBatch,
    revert: dict,
    **kwargs,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    # revert standardized locations and times for plotting
    batch = replace(
        batch,
        s_ctx=revert["s"](batch.s_ctx),
        s_test=revert["s"](batch.s_test),
        t_ctx=t_to_label(revert["t"](batch.t_ctx)),
        t_test=t_to_label(revert["t"](batch.t_test)),
    )
    f_pred, f_std = output.mu, output.std
    f_min = min(batch.f_ctx.min(), batch.f_test.min(), f_pred.min())
    f_max = max(batch.f_ctx.max(), batch.f_test.max(), f_pred.max())
    norm = Normalize(f_min, f_max)
    norm_std = Normalize(f_std.min(), f_std.max())
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    path = f"/tmp/era5_{step}_{datetime.now().isoformat()}.png"
    fig = batch.plot_2d(
        f_pred,
        f_std,
        cmap=cmap,
        # norm=norm,
        # norm_std=norm_std,
        **kwargs,
    )
    # TODO(danj): add tick labels for lat/lng
    fig.savefig(path)
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


def t_to_label(t: jax.Array):
    t = t.astype("M8[s]").astype(object)
    return np.vectorize(lambda x: x.strftime("%H:%M, %b %d"))(t)


if __name__ == "__main__":
    main()
