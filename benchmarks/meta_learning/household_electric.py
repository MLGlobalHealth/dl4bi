#!/usr/bin/env python3
import sys
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import jit, random, vmap
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.temporal import TemporalBatch, TemporalData
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
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 32,
    num_ctx_min: int = 32,
    num_ctx_max: int = 256,
    num_test: int = 256,
):
    train, valid, test = load_data(rng)
    x_train, t_train, f_train = train
    x_valid, t_valid, f_valid = valid
    x_test, t_test, f_test = test

    def build_dataloader(x, t, f):
        N, L = x.shape[0], num_ctx_max + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_b, rng_i, rng = random.split(rng, 3)
                rng_bs = random.split(rng_b, batch_size)
                idx = vmap(lambda rng: random.choice(rng, N, (L,), replace=False))(
                    rng_bs
                )  # [B, T]
                yield TemporalData(x[idx], t[idx], f[idx]).batch(
                    rng_i,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                )

        return dataloader

    def test_dataloader(rng: jax.Array):
        N = x_train.shape[0] + x_valid.shape[0]
        yield TemporalBatch(
            x_ctx=jnp.concat([x_train, x_valid], axis=0)[None, ...],
            t_ctx=jnp.concat([t_train, f_train], axis=0)[None, ...],
            f_ctx=jnp.concat([f_train, f_valid], axis=0)[None, ...],
            mask_ctx=jnp.ones((1, N), dtype=bool),
            x_test=x_test[None, ...],
            t_test=t_test[None, ...],
            f_test=f_test[None, ...],
            mask_test=jnp.ones((1, f_test.shape[0]), dtype=bool),
            inv_permute_idx=jnp.arange(N),
        )

    return (
        build_dataloader(x_train, t_train, f_train),
        build_dataloader(x_valid, t_valid, f_valid),
        test_dataloader,
    )


def load_data(rng: jax.Array):
    rng_valid, rng_test = random.split(rng)
    try:
        df = pd.read_csv("cache/household_power_consumption.txt", sep=";").dropna()
        df["dt"] = pd.to_datetime(df.Date + " " + df.Time, dayfirst=True)
        df["year"] = df.dt.dt.year
        df["month"] = df.dt.dt.month
        df["day"] = df.dt.dt.day
        df["hour"] = df.dt.dt.hour
        df["minute"] = df.dt.dt.minute
        df["t"] = (df.dt - df.dt.min()).dt.total_seconds()
        df = df.sort_values(by="t").reset_index(drop=True)
        df = df.drop(columns=["Date", "Time", "dt"])
    except FileNotFoundError:
        url = (
            "https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set"
        )
        msg = f"""
        1. Download the dataset here: {url}
        3. Move the file into the "cache" directory.
        """
        print(msg)
        sys.exit("Dataset not available.")
    N = df.shape[0]
    block_size = int(N * 0.1)
    df_train, df_valid = extract_temporal_block(rng_valid, df, block_size)
    df_train, df_test = extract_temporal_block(rng_test, df_train, block_size)
    df_train, df_valid, df_test = standardize_by_train(df_train, df_valid, df_test)
    t_cols = ["t"]
    f_cols = ["Global_active_power"]
    exclude_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    x_cols = list(set(df.columns) - set(t_cols + f_cols)) - set(exclude_cols)
    split_xtf = jit(lambda df: [df[c].values for c in [x_cols, t_cols, f_cols]])
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
    tx = StandardScaler()
    x_train = tx.fit_transform(df_train)
    x_valid = tx.transform(df_valid)
    x_test = tx.transform(df_test)
    df_train = pd.DataFrame(x_train, columns=cols)
    df_valid = pd.DataFrame(x_valid, columns=cols)
    df_test = pd.DataFrame(x_test, columns=cols)
    return df_train, df_valid, df_test


if __name__ == "__main__":
    main()
