#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler
from ucimlrepo import fetch_ucirepo

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.temporal import TemporalData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/household_electric", config_name="default", version_base=None)
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
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(
        rng_data, **cfg.data
    )
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
        return_state="best",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 32,
    num_ctx_min: int = 384,
    num_ctx_max: int = 384,
    num_test: int = 128,
):
    B = batch_size
    train, valid, test = load_data(rng)
    x_train, t_train, f_train = train
    x_valid, t_valid, f_valid = valid
    x_test, t_test, f_test = test

    def build_dataloader(x, t, f):
        N, L = x.shape[0], num_ctx_max + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_b, rng_i, rng = random.split(rng, 3)
                idx = random.choice(rng_b, N - L, (B, 1), replace=False)
                idx += jnp.arange(L)  # [B, L]
                yield TemporalData(x[idx], t[idx].squeeze(), f[idx]).batch(
                    rng_i,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                    forecast=True,
                    t_sorted=True,
                )

        return dataloader

    return (
        build_dataloader(x_train, t_train, f_train),
        build_dataloader(x_valid, t_valid, f_valid),
        build_dataloader(x_test, t_test, f_test),
    )


def load_data(rng: jax.Array):
    rng_valid, rng_test = random.split(rng)
    path = Path("cache/household_power_consumption.csv")
    try:
        df = pd.read_csv(path, na_values="?")
    except FileNotFoundError:
        df = fetch_ucirepo(id=235)["data"]["features"]
        df.to_csv(path, index=False)
    df["dt"] = pd.to_datetime(df.Date + " " + df.Time, dayfirst=True)
    df["year"] = df.dt.dt.year
    df["month"] = df.dt.dt.month
    df["day"] = df.dt.dt.day
    df["hour"] = df.dt.dt.hour
    df["day_of_week"] = df.dt.dt.day_of_week
    df["is_weekend"] = (df.dt.dt.day_of_week >= 5).astype(int)
    df["power"] = df.Global_active_power
    df["t"] = (df.dt - df.dt.min()).dt.total_seconds()
    df = df.sort_values(by="t").reset_index(drop=True)
    df = df.drop(columns=["Date", "Time", "dt"]).dropna()
    N = df.shape[0]
    block_size = int(N * 0.1)
    df_train, df_valid = extract_temporal_block(rng_valid, df, block_size)
    df_train, df_test = extract_temporal_block(rng_test, df_train, block_size)
    df_train, df_valid, df_test = standardize_by_train(df_train, df_valid, df_test)
    x_cols = ["year", "month", "day", "hour", "day_of_week", "is_weekend"]
    t_cols = ["t"]
    f_cols = ["power"]
    split_xtf = lambda df: [df[c].values for c in [x_cols, t_cols, f_cols]]
    return split_xtf(df_train), split_xtf(df_valid), split_xtf(df_test)


def extract_temporal_block(
    rng: jax.Array,
    df: pd.DataFrame,
    block_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts a contiguous temporal block.

    .. note::
        Assumes the data is temporally ordered.
    """
    N = df.shape[0]
    i = random.choice(rng, N, (1,)).item()
    df_block = df.iloc[i : i + block_size]
    df_sans_block = df.drop(df.index[i : i + block_size]).reset_index(drop=True)
    return df_sans_block, df_block


def standardize_by_train(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = df_train.columns
    tx = MinMaxScaler()
    x_train = tx.fit_transform(df_train)
    x_valid = tx.transform(df_valid)
    x_test = tx.transform(df_test)
    df_train = pd.DataFrame(x_train, columns=cols)
    df_valid = pd.DataFrame(x_valid, columns=cols)
    df_test = pd.DataFrame(x_test, columns=cols)
    return df_train, df_valid, df_test


if __name__ == "__main__":
    main()
