#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.utils import cfg_to_run_name
from dl4bi.regression.steps import RegressionBatch


@hydra.main("configs/dengue", config_name="default", version_base=None)
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
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(**cfg.data)
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
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))
    # TODO(danj): load test batches and plot a sample
    # TODO(danj): partition dataset into pre-2018 and post?


def build_dataloaders(
    num_ctx: int = 384,
    num_test: int = 30,
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
):
    df_train, df_valid, df_test = load_data(pct_train, pct_valid, pct_test)

    def build_dataloader(df: pd.DataFrame):
        N, L = len(df), num_ctx + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng = random.split(rng)
                idx = random.choice(rng_i, N - L, (1,), replace=False).item()
                d = df.iloc[idx : idx + L].values.T  # [num districts, L]
                x, y = d[:, :num_ctx], d[:, num_ctx:]
                yield RegressionBatch(x, y)

        return dataloader

    return (
        build_dataloader(df_train),
        build_dataloader(df_valid),
        build_dataloader(df_test),
    )


def load_data(
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path("cache/dengue.parquet")
    path = Path("~/scratch/daily_d_ts.parquet")
    features = ["date", "district", "n"]
    df = pd.read_parquet(path)[features]
    df = df.set_index("date").sort_index()
    # TODO(danj): separate model for later years?
    df = df[df.index < "2017-01-01"]
    idx = pd.date_range(df.index.min(), df.index.max())
    df = df.groupby("district").apply(forward_fill, idx)
    df.index = df.index.droplevel(0)
    # index: date, columns: districts, values: n
    df = df.pivot(columns="district", values="n")
    N = df.shape[0]
    num_train, num_valid, num_test = map(
        lambda pct: int(N * pct), (pct_train, pct_valid, pct_test)
    )
    df_train, df_test = df[:-num_test], df[-num_test:]
    df_train, df_valid = df_train[:-num_valid], df_train[-num_valid:]
    df_train = df_train[:num_train]
    print(f"Train N: {num_train}, Valid N: {num_valid}, Test N: {num_test}")
    return df_train, df_valid, df_test
    # TODO(danj): standardize by district?
    # return standardize_by_train(df_train, df_valid, df_test)


# TODO(danj): think of a better filling strategy
def forward_fill(df, idx):
    district = df["district"].iloc[0]
    df = df.reindex(idx)
    num_na = df["n"].isna().sum()
    df["district"] = district
    df["n"] = df["n"].ffill()
    print(f"forward filled {num_na} 'n' values for {district}")
    return df


def standardize_by_train(
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    num_feats = ["n"]
    tx = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    x_train = tx.fit_transform(df_train)
    x_valid = tx.transform(df_valid)
    x_test = tx.transform(df_test)
    cols = tx.get_feature_names_out().tolist()
    df_train = pd.DataFrame(x_train, columns=cols).infer_objects()
    df_valid = pd.DataFrame(x_valid, columns=cols).infer_objects()
    df_test = pd.DataFrame(x_test, columns=cols).infer_objects()
    return df_train, df_valid, df_test, tx


if __name__ == "__main__":
    main()
