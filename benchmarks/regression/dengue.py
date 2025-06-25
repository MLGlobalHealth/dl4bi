#!/usr/bin/env python3
import pickle
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import wandb
from flax.core.frozen_dict import FrozenDict
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from dl4bi.core.train import (
    evaluate,
    infer,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.utils import cfg_to_run_name


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
    train_dataloader, valid_dataloader, test_dataloader, tx = build_dataloaders(
        **cfg.data
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
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))
    batches = test_dataloader(rng_test)
    invert_district = tx.transformers_[0][1].inverse_transform
    t_min = tx.transformers_[1][1].data_min_[-1]
    t_max = tx.transformers_[1][1].data_max_[-1]
    t_diff = t_max - t_min
    start_date = pd.Timestamp("1999-12-31", tz="UTC")
    invert_t = lambda t: start_date + pd.to_timedelta(
        np.array(t) * t_diff + t_min, unit="s"
    )
    lambdas, districts, ts_ctx, ts_test, fs_ctx, fs_test = [], [], [], [], [], []
    for _ in range(cfg.infer_num_steps):
        rng_i, rng = random.split(rng)
        batch = next(batches)
        output = infer(rng_i, state, batch)
        d_ctx, d_test, t_ctx, t_test, f_ctx, f_test = (
            batch.ctx["x_ctx"][..., 0],
            batch.test["x_test"][..., 0],
            batch.ctx["t_ctx"],
            batch.test["t_test"],
            batch.ctx["f_ctx"],
            batch.test["f_test"],
        )
        ds_ctx += [
            np.array([invert_district(d.reshape(-1, 1)).flatten() for d in d_ctx])
        ]
        ds_test += [
            np.array([invert_district(d.reshape(-1, 1)).flatten() for d in d_test])
        ]
        lambdas += [np.array(output.lam)]
        ts_ctx += [np.array([invert_t(t_i.flatten()) for t_i in t_ctx])]
        ts_test += [np.array([invert_t(t_i.flatten()) for t_i in t_test])]
        fs_ctx += [np.array(f_ctx)]
        fs_test += [np.array(f_test)]
    with open("/tmp/results.pkl", "wb") as fp:
        pickle.dump(
            {
                "lambda": stack(lambdas),
                "t_ctx": stack(ts_ctx),
                "t_test": stack(ts_test),
                "f_ctx": stack(fs_ctx),
                "f_test": stack(fs_test),
                "district_ctx": stack(districts_ctx),
                "district_test": stack(districts_test),
            },
            fp,
        )


def stack(x):
    x = np.array(x)
    return x.reshape(-1, *x.shape[2:])


def build_dataloaders(
    batch_size: int = 32,
    num_ctx_min: int = 2250,  # (24 + 1) * 90 => ~3 months
    num_ctx_max: int = 2250,
    num_test: int = 350,  # (24 + 1) * 14 => 2 weeks
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
):
    B = batch_size
    train, valid, test, tx = load_data(pct_train, pct_valid, pct_test)

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

    return (
        build_dataloader(*train),
        build_dataloader(*valid),
        build_dataloader(*test),
        tx,
    )


def load_data(
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
):
    path = Path("cache/dengue.parquet")
    path = Path("~/scratch/daily_d_ts.parquet")
    features = ["date", "district", "n"]
    df = pd.read_parquet(path)[features]
    df = df.set_index("date").sort_index()
    idx = pd.date_range(df.index.min(), df.index.max())
    df = df.groupby("district").apply(forward_fill, idx, include_groups=True)
    # TODO(danj): this data is currently in multiindex [district, date]
    # 1. need to standardize this data (by district?)
    # 2. need to create a sampler/dataloader
    N = df.shape[0]
    num_train, num_valid, num_test = map(
        lambda pct: int(N * pct), (pct_train, pct_valid, pct_test)
    )
    df_train, df_test = df[:-num_test], df[-num_test:]
    df_train, df_valid = df_train[:-num_valid], df_train[-num_valid:]
    df_train = df_train[:num_train]
    return standardize_by_train(df_train, df_valid, df_test)


def forward_fill(df, idx):
    df = df.reindex(idx)
    df["district"] = df["district"].iloc[0]
    df["n"] = df["n"].ffill()
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
