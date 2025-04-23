#!/usr/bin/env python3
import os
from pathlib import Path

import cdsapi
import hydra
import jax
import wandb
import xarray as xr
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
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
    rng_data, rng_train, rng_test = random.split(rng, 3)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(rng_data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


# TODO(danj): update
def build_dataloaders(
    batch_size: int = 16,
    num_ctx_min: int = 56,  # ~5% of 1,125
    num_ctx_max: int = 287,  # ~25.5% of 1,125
    num_test: int = 225,  # 287 + 225 = 512 = L
):
    B = batch_size
    # TODO(danj):
    # 1. Sample [15, 15, 4] from data
    # 2. Sample 5 frames, each 6 hours apart
    # 3. Randomly mask and predict 6th frame
    ds_train = load_data("central_europe")
    ds_test_1 = load_data("western_europe")
    ds_test_2 = load_data("northern_europe")

    def build_dataloader(dataset, is_callback: bool = False):
        N = dataset.shape[0]

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_b, rng = random.split(rng, 3)
                batch_idx = random.choice(rng_i, N, (B,), replace=False)
                f = dataset[batch_idx]
                d = SpatiotemporalData(x=None, s=s, f=f)
                yield d.batch(
                    rng_b,
                    num_ctx_min,
                    num_ctx_max,
                    32 * 32 if is_callback else num_test,
                    test_includes_ctx=True,
                )

        return dataloader

    return (
        build_dataloader(ds_train),
        build_dataloader(ds_test_1),
        build_dataloader(ds_test_2),
    )


def load_data(region: str = "central_europe"):
    download_if_not_cached()
    ds = xr.open_mfdataset(f"cache/era5/{region}/2019_*.nc", combine="by_coords")
    df = ds.to_dataframe()[["t2m", "z"]].reset_index()
    return df


def download_if_not_cached():
    client = cdsapi.Client()
    variables = ["2m_temperature", "geopotential"]
    times = [f"{t:02d}:00" for t in range(24)]
    central_europe = [53, 8, 42, 28]  # N, W, S, E
    western_europe = [53, -4, 42, 8]
    northern_europe = [62, 8, 53, 28]
    grid = [0.5, 0.5]

    for region, area in [
        ("central_europe", central_europe),
        ("western_europe", western_europe),
        ("northern_europe", northern_europe),
    ]:
        os.makedirs(f"cache/era5/{region}", exist_ok=True)
        for month in [f"{m:02d}" for m in range(1, 13)]:
            target = Path(f"cache/era5/{region}/2019_{month}.nc")
            if not target.exists():
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


if __name__ == "__main__":
    main()
