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
        **cfg.data.callback_dataloader,
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


def dataloader(
    rng: jax.Array,
    ds: xr.Dataset,
    img_size: int = 32,
    num_ctx_min_per_t: int = 50,
    num_ctx_max_per_t: int = 256,
    num_test: int = 1024,
    num_t: int = 5,
    t_interval: int = 2,  # every hour (values are every half hour)
    batch_size: int = 16,
    num_batches_per_subset: int = 50,
    is_callback: bool = False,
    revert: Optional[Callable] = None,
):
    R, H, W, S = ds.region.size, ds.i.size, ds.j.size, img_size
    minval, maxval = jnp.array([0, 0]), jnp.array([H - S, W - S])
    while True:
        rng_r, rng_t, rng_pos, rng = random.split(rng, 4)
        r = random.choice(rng_r, R, (1,)).item()
        t_start = random.choice(rng_t, 48, (1,)).item()
        t_mask = (ds.half_hour_of_day + t_start) % t_interval == 0
        i, j = random.randint(rng_pos, (2,), minval, maxval)
        ds_subset = ds.isel(
            region=r,
            i=slice(i, i + S),
            j=slice(j, j + S),
            time=t_mask,
        )
        precip_std = ds_subset.precip_log1p_standardized
        subset = SpatiotemporalData(
            x=ds_subset.half_hour_of_day_normalized.broadcast_like(precip_std).values[
                ..., None
            ],
            s=ds_subset.spherical_coords.broadcast_like(precip_std).values,
            t=ds_subset.half_hour_since_start_standardized.values,
            f=ds_subset.precip_log1p_standardized.values[..., None],
        )
        # create a number of batches from this filtered subset
        for _ in range(num_batches_per_subset):
            rng_b, rng = random.split(rng)
            batch = subset.batch(
                rng=rng_b,
                num_t=num_t,
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
    train_years: Sequence[int] = [2018, 2019, 2020, 2021],
    valid_years: Sequence[int] = [2023],
    test_years: Sequence[int] = [2023],
):
    ds = xr.open_zarr("cache/nairobi_rainfall", consolidated=True)
    half_hour_of_day = ds.time.dt.hour.data * 2 + ds.time.dt.minute.data // 30
    ds = ds.assign_coords({"half_hour_of_day": ("time", half_hour_of_day)})
    # NOTE: overwrite to save space (this is huge)
    ds["precipitation"] = xr.ufuncs.log1p(ds.precipitation)
    ds_train, ds_valid, ds_test = split_train_valid_test(
        ds, train_years, valid_years, test_years
    )
    print("\nStandardizing data based on training set; may take a bit...")
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
    train_years: Sequence[int] = [2018, 2019, 2020, 2021],
    valid_years: Sequence[int] = [2023],
    test_years: Sequence[int] = [2023],
):
    extract = lambda years: ds.isel(time=ds.time.dt.year.isin(years))
    return extract(train_years), extract(valid_years), extract(test_years)


def standardize_using_train(
    ds_train: xr.Dataset,
    ds_valid: xr.Dataset,
    ds_test: xr.Dataset,
):
    t_min = ds_train.time.min().values

    def _half_hour(ds: xr.Dataset):
        return (ds.time - t_min) / np.timedelta64(30, "m")

    def _mu_std(arr: xr.DataArray):
        return arr.mean().compute().item(), arr.std().compute().item()

    def _stdize(arr, mu, std):
        return ((arr - mu) / std).data.astype("float32")

    precip_mu, precip_std = _mu_std(ds_train.precipitation)
    half_hour_mu, half_hour_std = _mu_std(_half_hour(ds_train))

    def standardize(ds: xr.Dataset):
        return ds.assign(
            {
                "precip_log1p_standardized": (
                    ("region", "time", "i", "j"),
                    _stdize(ds.precipitation, precip_mu, precip_std),
                ),
                "half_hour_since_start_standardized": (
                    "time",
                    _stdize(_half_hour(ds), half_hour_mu, half_hour_std),
                ),
                "half_hour_of_day_normalized": (
                    "time",
                    (ds.half_hour_of_day / 47.0).data.astype("float32"),
                ),
            }
        )

    def revert_t(t: jax.Array):
        half_hours = np.rint(t * half_hour_std + half_hour_mu)
        return t_min + (half_hours * np.timedelta64(30, "m"))

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
