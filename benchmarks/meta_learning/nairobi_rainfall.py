#!/usr/bin/env python3
import os
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence

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


@hydra.main("configs/nairobi_rainfall", config_name="default", version_base=None)
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
        cfg.data.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


# TODO(danj): implement
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
    train_years: Sequence[int],
    valid_years: Sequence[int],
    test_years: Sequence[int],
):
    ds = xr.open_zarr("cache/nairobi_rainfall", consolidated=True)
    ds_train, ds_valid, ds_test = split_train_valid_test(
        ds, train_years, valid_years, test_years
    )
    return standardize_using_train(ds_train, ds_valid, ds_test)


def download(bucket_name):  # bucket_name is private for now
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip "folders"
        local_path = os.path.join("cache/nairobi_rainfall", blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def split_train_valid_test(
    ds: xr.Dataset,
    train_years: Sequence[int],
    valid_years: Sequence[int],
    test_years: Sequence[int],
):
    # TODO(danj): implement
    return ds_train, ds_valid, ds_test


def standardize_using_train(
    ds_train: xr.Dataset,
    ds_valid: xr.Dataset,
    ds_test: xr.Dataset,
):
    t_min = ds_train.time.min().values

    def _half_hour(ds: xr.Dataset):
        return (ds.time - t_min) / np.timedelta64(30, "m")

    def _mu_std(arr: xr.DataArray):
        return arr.mean().item(), arr.std().item()

    def _stdize(arr, mu, std):
        return ((arr - mu) / std).data.astype("float32")

    half_hour_mu, half_hour_std = _mu_std(_half_hour(ds_train))

    def standardize(ds: xr.Dataset):
        return ds.assign(
            {
                "half_hour_since_start_standardized": (
                    "time",
                    _stdize(_half_hour(ds), half_hour_mu, half_hour_std),
                ),
                "hour_of_day_normalized": (
                    "time",
                    (ds.time.dt.hour / 23.0).data.astype("float32"),
                ),
            }
        )

    def revert_t(t: jax.Array):
        half_hours = (
            np.rint(t * half_hour_std + half_hour_mu)
            .astype(int)
            .astype(np.timedelta64(30, "m"))
        )
        return t_min + half_hours

    return (
        standardize(ds_train),
        standardize(ds_valid),
        standardize(ds_test),
        {"t": revert_t},
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
        t_ctx=t_to_label(revert["t"](batch.t_ctx)),
        t_test=t_to_label(revert["t"](batch.t_test)),
    )
    f_pred, f_std = output.mu, output.std
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    path = f"/tmp/nairobi_rainfall_{step}_{datetime.now().isoformat()}.png"
    fig = batch.plot_2d(
        f_pred,
        f_std,
        cmap=cmap,
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
