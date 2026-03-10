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
    load_ckpt,
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
    ds_train, ds_valid, ds_test, revert = load_data(**cfg.data.splits)
    train_dataloader = partial(dataloader, ds=ds_train, **cfg.data.train_dataloader)
    valid_dataloader = partial(dataloader, ds=ds_valid, **cfg.data.valid_dataloader)
    test_dataloader = partial(dataloader, ds=ds_test, **cfg.data.test_dataloader)
    callback_dataloader = partial(
        dataloader,
        ds=ds_valid,
        is_callback=True,
        revert=revert,
        **cfg.data.valid_dataloader,
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    if cfg.evaluate_only:
        path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
        state, _ = load_ckpt(path.with_suffix(".ckpt"), cfg)
        metrics = evaluate(
            rng_test,
            state,
            model.valid_step,
            test_dataloader,
            cfg.data.test_num_steps,
        )
        wandb.log({f"Test {m}": v for m, v in metrics.items()})
        return
    clbk = Callback(plot, cfg.data.plot_interval)
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
        cfg.data.test_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def dataloader(
    rng: jax.Array,
    ds: xr.Dataset,
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
    lat_uniq, lng_uniq = ds.latitude.data, ds.longitude.data
    lat_choices = lat_uniq[lat_uniq <= lat_uniq.max() - H_deg]
    lng_choices = lng_uniq[lng_uniq <= lng_uniq.max() - W_deg]
    while True:
        # filter to random starting time and lat/lng block
        rng_t, rng_lat, rng_lng, rng_b, rng = random.split(rng, 5)
        hr_start = random.choice(rng_t, T_hrs_delta, (1,)).item()
        lat_start = random.choice(rng_lat, lat_choices, (1,)).item()
        lng_start = random.choice(rng_lng, lng_choices, (1,)).item()
        time_idx = (ds.hour_of_day + hr_start) % T_hrs_delta == 0
        ds_subset = ds.sel(
            time=ds.time[time_idx],
            # add or subtract 1e-6 because upper bounds are exclusive
            latitude=slice(lat_start + H_deg, lat_start + 1e-6),  # lats are decreasing
            longitude=slice(lng_start, lng_start + W_deg - 1e-6),  # lngs are increasing
        )
        elev_std = ds_subset.elevation_standardized
        subset = SpatiotemporalData(
            x=jnp.stack(
                [
                    elev_std.values,
                    ds_subset.hour_of_day_normalized.broadcast_like(elev_std).values,
                ],
                axis=-1,
            ),
            s=jnp.stack(
                [
                    ds_subset.latitude_standardized.broadcast_like(elev_std).values,
                    ds_subset.longitude_standardized.broadcast_like(elev_std).values,
                ],
                axis=-1,
            ),
            t=ds_subset.hour_since_start_standardized.values,
            f=ds_subset.temperature_standardized.broadcast_like(elev_std).values[
                ..., None
            ],
        )
        # create a number of batches from this filtered subset
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
    ds_train = load_cached(train_region)
    ds_valid = load_cached(valid_region)
    ds_test = load_cached(test_region)
    return standardize_using_train(ds_train, ds_valid, ds_test)


def load_cached(region: str = "central_europe") -> xr.Dataset:
    ds = xr.open_mfdataset(f"cache/era5/{region}/2019_*.nc", combine="by_coords")
    ds = ds.rename({"valid_time": "time", "z": "elevation", "t2m": "temperature"})
    ds = ds.assign_coords({"hour_of_day": ("time", ds.time.dt.hour.data)})
    return ds.load()


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
    ds_train: xr.Dataset,
    ds_valid: xr.Dataset,
    ds_test: xr.Dataset,
):
    t_min = ds_train.time.min().values

    def _hour(ds: xr.Dataset):
        return (ds.time - t_min) / np.timedelta64(1, "h")

    def _mu_std(arr: xr.DataArray):
        return arr.mean().item(), arr.std().item()

    def _stdize(arr, mu, std):
        return ((arr - mu) / std).data.astype("float32")

    hour_mu, hour_std = _mu_std(_hour(ds_train))
    lat_mu, lat_std = _mu_std(ds_train.latitude)
    lng_mu, lng_std = _mu_std(ds_train.longitude)
    elev_mu, elev_std = _mu_std(ds_train.elevation)
    temp_mu, temp_std = _mu_std(ds_train.temperature)

    def standardize(ds: xr.Dataset):
        return ds.assign(
            {
                "hour_since_start_standardized": (
                    "time",
                    _stdize(_hour(ds), hour_mu, hour_std),
                ),
                "hour_of_day_normalized": (
                    "time",
                    (ds.time.dt.hour / 23.0).data.astype("float32"),
                ),
                "latitude_standardized": (
                    "latitude",
                    _stdize(ds.latitude, lat_mu, lat_std),
                ),
                "longitude_standardized": (
                    "longitude",
                    _stdize(ds.longitude, lng_mu, lng_std),
                ),
                "elevation_standardized": (
                    ("time", "latitude", "longitude"),
                    _stdize(ds.elevation, elev_mu, elev_std),
                ),
                "temperature_standardized": (
                    ("time", "latitude", "longitude"),
                    _stdize(ds.temperature, temp_mu, temp_std),
                ),
            }
        )

    def revert_t(t: jax.Array):
        hours = np.rint(t * hour_std + hour_mu).astype(int).astype("timedelta64[h]")
        return t_min + hours

    def revert_s(s: jax.Array):  # s: [T, H * W, 2]
        std = jnp.array([lat_std, lng_std]).reshape(1, 1, 2)
        mu = jnp.array([lat_mu, lng_mu]).reshape(1, 1, 2)
        return jnp.round(s * std + mu, decimals=1)

    return (
        standardize(ds_train),
        standardize(ds_valid),
        standardize(ds_test),
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
