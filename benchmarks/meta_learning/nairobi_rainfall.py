#!/usr/bin/env python3
import os
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable

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
    ds_train, ds_test = load_data()
    train_dataloader = partial(dataloader, ds=ds_train, **cfg.data.train_dataloader)
    valid_dataloader = partial(dataloader, ds=ds_test, **cfg.data.valid_dataloader)
    test_dataloader = partial(dataloader, ds=ds_test, **cfg.data.test_dataloader)
    callback_dataloader = partial(
        dataloader,
        ds=ds_test,
        is_callback=True,
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
    chunk_size: int = 1024,
    num_batches_per_subset: int = 50,
    is_callback: bool = False,
):
    R, T, H, W, S = ds.region.size, ds.t.size, ds.i.size, ds.j.size, img_size
    minval, maxval = jnp.array([0, 0]), jnp.array([H - S, W - S])

    def revert_t(t: jax.Array):
        return ds.t_min.values + np.array(t) * np.timedelta64(1, "h")

    while True:
        rng_t, rng_pos, rng = random.split(rng, 3)
        t_start = random.choice(rng_t, T - (chunk_size * t_interval), (1,)).item()
        i, j = random.randint(rng_pos, (2,), minval, maxval)
        ds_subset = ds.isel(
            i=slice(i, i + S),
            j=slice(j, j + S),
            time=slice(t_start, t_start + (chunk_size * t_interval), t_interval),
        ).load()
        # create a number of batches from this filtered subset
        for _ in range(num_batches_per_subset):
            rng_r, rng_b, rng = random.split(rng, 3)
            r = random.choice(rng_r, R, (1,)).item()
            ds_subset_r = ds_subset.sel(region=r)
            precip = ds_subset_r.precip_log1p
            subset = SpatiotemporalData(
                x=None,
                # x=jnp.concat(
                #     [
                #         ds_subset_r.t_day.broadcast_like(precip).values[..., None],
                #         ds_subset_r.t_year.broadcast_like(precip).values[..., None],
                #         ds_subset_r.region.broadcast_like(precip).values[..., None],
                #         ds_subset_r.spherical_coords.broadcast_like(precip).values,
                #     ],
                #     axis=-1,
                # ),
                s=jnp.concat(
                    [
                        ds_subset_r.i.broadcast_like(precip).values[..., None] / H,
                        ds_subset_r.j.broadcast_like(precip).values[..., None] / W,
                    ],
                    axis=-1,
                ),
                t=ds_subset_r.t.values,
                f=ds_subset_r.precip_log1p.values[..., None],
            )
            # try num_tries to get min_pct rainfall
            num_tries, min_pct = 10, 0.02
            best_batch, best_pct = None, 0.0
            for _ in range(num_tries):
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
                pct = batch.f_ctx.sum() / batch.f_ctx.size
                if pct >= best_pct:
                    best_batch = batch
                    best_pct = pct
                if best_pct > min_pct:
                    break
            yield (best_batch, revert_t) if is_callback else best_batch


def load_data():
    path = Path("cache/nairobi_rainfall")
    if not (path / "train.zarr").exists():
        return download_and_preprocess()
    return (
        xr.open_zarr(path / "train.zarr", consolidated=True),
        xr.open_zarr(path / "test.zarr", consolidated=True),
    )


def download_and_preprocess():
    path = "cache/nairobi_rainfall/raw.zarr"
    if not Path(path).exists():
        raise Exception("Missing data! See 'download' function in this file!")
    ds = xr.open_zarr("cache/nairobi_rainfall/raw.zarr", consolidated=True)
    ds_train, ds_test = preprocess(ds)
    ds_train.to_zarr(
        "cache/nairobi_rainfall/train.zarr",
        mode="w",
        consolidated=True,
        zarr_version=2,
    )
    ds_test.to_zarr(
        "cache/nairobi_rainfall/test.zarr",
        mode="w",
        consolidated=True,
        zarr_version=2,
    )
    return ds_train, ds_test


def download(bucket_name):  # bucket_name is private for now
    # pip install google-cloud-storage
    from google.cloud import storage

    client = storage.Client.create_anonymous_client()
    blobs = client.list_blobs(bucket_name)
    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip "folders"
        local_path = os.path.join("cache/nairobi_rainfall/raw.zarr", blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


def preprocess(ds: xr.Dataset, train_years=[2018, 2019, 2020, 2021], test_years=[2023]):
    t_min = np.datetime64(f"{train_years[0]}-01-01", "s")
    ds["precip_log1p"] = xr.ufuncs.log1p(ds.precipitation)
    ds = ds.assign_coords({"t": (ds.time - t_min) / np.timedelta64(1, "h")})
    ds = ds.assign_coords({"t_day": ds.time.dt.hour / 23.0})
    ds = ds.assign_coords({"t_year": ds.time.dt.dayofyear / 365})
    ds["t_min"] = t_min
    ds = ds[["precip_log1p", "t", "t_day", "t_year", "spherical_coords", "t_min"]]
    extract = lambda years: ds.isel(time=ds.time.dt.year.isin(years))
    return extract(train_years), extract(test_years)


def plot(
    step: int,
    rng_step: int,
    state: TrainState,
    batch: SpatiotemporalBatch,
    revert_t: Callable,
    **kwargs,
):
    """Logs `num_plots` from the given batch for 2D GPs."""
    rng_dropout, rng_extra = random.split(rng_step)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        rngs={"dropout": rng_dropout, "extra": rng_extra},
    )
    batch = replace(
        batch,
        t_ctx=t_to_label(revert_t(batch.t_ctx)),
        t_test=t_to_label(revert_t(batch.t_test)),
    )
    f_pred, f_std = output.mu, output.std
    f_min = min(batch.f_ctx.min(), batch.f_test.min(), f_pred.min())
    f_max = max(batch.f_ctx.max(), batch.f_test.max(), f_pred.max())
    norm = Normalize(f_min, f_max)
    norm_std = Normalize(f_std.min(), f_std.max())
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    path = f"/tmp/nairobi_rainfall_{step}_{datetime.now().isoformat()}.png"
    fig = batch.plot_2d(
        f_pred,
        f_std,
        cmap=cmap,
        norm=norm,
        norm_std=norm_std,
        **kwargs,
    )
    # TODO(danj): add tick labels for lat/lng
    fig.savefig(path)
    plt.close(fig)
    wandb.log({f"Step {step}": wandb.Image(path)})


def t_to_label(t: np.ndarray):
    t = t.astype("M8[s]").astype(object)
    return np.vectorize(lambda x: x.strftime("%H:%M, %b %d"))(t)


if __name__ == "__main__":
    main()
