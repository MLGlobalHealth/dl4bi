#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import wandb
from jax import random
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from dl4bi.core import Whitener
from dl4bi.meta_regression.train_utils import (
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    save_ckpt,
    train,
)


# TODO(danj): verify data prep
# TODO(danj): determine matern_3_2 priors for each dataset
@hydra.main("configs/uci", config_name="default", version_base=None)
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
    rng_dataloaders, rng_train, rng_test = random.split(rng, 3)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(
        rng_dataloaders,
        cfg.data,
        cfg.kernel,
    )
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
    )
    metrics = evaluate(rng_test, state, test_dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(rng: jax.Array, data: DictConfig, kernel: DictConfig):
    s_train, f_train, s_valid, f_valid, s_test, f_test = load_dataset(rng, data.name)
    N_train, N_valid, N_test = s_train.shape[0], s_valid.shape[0], s_test.shape[0]
    B, D = data.batch_size, s_train.shape[-1]

    def train_dataloader(rng: jax.Array):
        L_batch = data.num_ctx.max + data.num_test
        valid_lens_test = jnp.repeat(L_batch, B)
        gp = instantiate(kernel)
        while True:
            rng_s, rng_gp, rng_valid, rng = random.split(rng, 4)
            s = random.normal(rng_s, (L_batch, D))
            f, *_ = gp.simulate(rng_gp, s, B)
            s = jnp.repeat(s[None, ...], B, axis=0)
            valid_lens_ctx = random.randint(
                rng_valid,
                (B,),
                data.num_ctx.min,
                data.num_ctx.max,
            )
            yield (s, f, valid_lens_ctx, s, f, valid_lens_test)

    def valid_dataloader(rng: jax.Array):
        yield (
            s_train[None, ...],
            f_train[None, ...],
            jnp.array([N_train]),
            s_valid[None, ...],
            f_valid[None, ...],
            jnp.array([N_valid]),
        )

    def test_dataloader(rng: jax.Array):
        yield (
            jnp.vstack([s_train, s_valid])[None, ...],
            jnp.vstack([f_train, f_valid])[None, ...],
            jnp.array([N_train + N_valid]),
            s_test[None, ...],
            f_test[None, ...],
            jnp.array([N_test]),
        )

    return train_dataloader, valid_dataloader, test_dataloader


def load_dataset(rng: jax.Array, name: str):
    target = {"bike": "count"}[name]
    path = Path(f"cache/uci/{name}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
        return preprocess(rng, df, target)
    match name:
        case "bike":
            data = fetch_ucirepo(id=275)
            df = data.data.features.drop("dteday", axis=1)
            df["count"] = data.data.targets["cnt"]
        case _:
            raise Exception(f"Invalid dataset {name}!")
    df.to_csv(f"cache/uci/{name}.csv", index=False)
    return preprocess(rng, df, target)


def preprocess(rng: jax.Array, df: pd.DataFrame, target: str):
    """Whitens and standardizes the data."""
    N = df.shape[0]
    features = list(set(df.columns) - {target})
    N_train, N_test = int(N * 4 / 9), int(N * 3 / 9)  # see 1M GP paper
    permute_idx = random.choice(rng, N, (N,), replace=False)
    df = df.iloc[permute_idx]
    df_train, df_valid, df_test = df[:N_train], df[N_train:-N_test], df[-N_test:]
    whitener = Pipeline([("whitener", Whitener()), ("standardizer", StandardScaler())])
    standardizer = StandardScaler()
    s_train = whitener.fit_transform(df_train[features].values)
    s_valid = whitener.fit_transform(df_valid[features].values)
    s_test = whitener.transform(df_test[features].values)
    f_train = standardizer.fit_transform(df_train[[target]].values)
    f_valid = standardizer.fit_transform(df_valid[[target]].values)
    f_test = standardizer.transform(df_test[[target]].values)
    return s_train, f_train, s_valid, f_valid, s_test, f_test


if __name__ == "__main__":
    main()
