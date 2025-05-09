#!/usr/bin/env python3
import sys
from io import StringIO
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from flax.core.frozen_dict import FrozenDict
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/beijing_air_quality", config_name="default", version_base=None)
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
        cfg.test_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 32,
    num_ctx_min: int = 624,  # 48 hours * 12 locations + 12 buffer
    num_ctx_max: int = 624,
    num_test: int = 84,  # 6 hours * 12 locations + 12 buffer
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
):
    B = batch_size
    train, valid, test = load_data(rng, pct_train, pct_valid, pct_test)

    def build_dataloader(x: jax.Array, s: jax.Array, t: jax.Array, f: jax.Array):
        N, L = x.shape[0], num_ctx_max + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_b, rng = random.split(rng, 3)
                idx = random.choice(rng_i, N - L, (B, 1), replace=False)
                idx += jnp.arange(L)  # [B, L]
                feature_groups = FrozenDict({"x": x[idx], "s": s[idx], "t": t[idx]})
                yield TabularData(feature_groups, f[idx]).batch(
                    rng_b,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                    forecast=True,
                    t_sorted=True,
                )

        return dataloader

    return build_dataloader(*train), build_dataloader(*valid), build_dataloader(*test)


def load_data(
    rng: jax.Array,
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
):
    rng_valid, rng_test = random.split(rng)
    dir = Path("cache/beijing_air_quality")
    try:
        df = pd.concat([pd.read_csv(p) for p in dir.glob("*.csv")], ignore_index=True)
        df_coords = load_coords()
        df = df.merge(df_coords, on="station", how="left")
        df = df.drop(columns=["station", "No"]).dropna()
    except Exception:
        url = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
        msg = f"""
        1. Download the dataset here: {url}
        2. Unzip the dataset
        3. Move files from the PRSA_* directory into "{dir}"
        """
        print(msg)
        sys.exit("Dataset not available.")
    df["t"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["day_of_week"] = df.t.dt.day_of_week
    df["is_weekend"] = (df.t.dt.day_of_week >= 5).astype(int)
    df["t"] = (df.t - df.t.min()).dt.total_seconds()
    df = df.sort_values(by="t").reset_index(drop=True)
    x_cols = [
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "is_weekend",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "wd",
        "WSPM",
    ]
    s_cols = ["Latitude", "Longitude"]
    t_cols = ["t"]
    f_cols = ["PM2.5", "PM10"]
    df = df[x_cols + s_cols + t_cols + f_cols]
    N = df.shape[0]
    num_train, num_valid, num_test = map(
        lambda pct: int(N * pct), (pct_train, pct_valid, pct_test)
    )
    df_train, df_test = df[:-num_test], df[-num_test:]
    df_train, df_valid = df_train[:-num_valid], df_train[-num_valid:]
    df_train = df_train[:num_train]
    df_train, df_valid, df_test = standardize_by_train(df_train, df_valid, df_test)
    x_cols.remove("wd")
    x_cols += [c for c in df_train.columns if c.startswith("wd_")]
    split_xstf = lambda df: [df[c].values for c in [x_cols, s_cols, t_cols, f_cols]]
    return split_xstf(df_train), split_xstf(df_valid), split_xstf(df_test)


def load_coords():
    return pd.read_csv(
        StringIO("""
station,Latitude,Longitude
Aotizhongxin,39.982,116.397
Changping,40.217,116.230
Dingling,40.292,116.220
Dongsi,39.929,116.417
Guanyuan,39.929,116.339
Gucheng,39.914,116.184
Huairou,40.328,116.628
Nongzhanguan,39.937,116.461
Shunyi,40.127,116.655
Tiantan,39.886,116.407
Wanliu,39.987,116.287
Wanshouxigong,39.878,116.352
    """)
    )


def standardize_by_train(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    num_feats = list(set(df_train.columns) - {"wd"})
    cat_feats = ["wd"]  # wind direction
    tx = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), num_feats),
            ("cat", OneHotEncoder(sparse_output=False), cat_feats),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    x_train = tx.fit_transform(df_train)
    x_valid = tx.transform(df_valid)
    x_test = tx.transform(df_test)
    cols = tx.get_feature_names_out().tolist()
    df_train = pd.DataFrame(x_train, columns=cols)
    df_valid = pd.DataFrame(x_valid, columns=cols)
    df_test = pd.DataFrame(x_test, columns=cols)
    return df_train, df_valid, df_test


if __name__ == "__main__":
    main()
