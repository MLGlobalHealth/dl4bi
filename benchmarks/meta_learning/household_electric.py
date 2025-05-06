#!/usr/bin/env python3
import sys
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import random, vmap
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import (
    TemporalBatch,
    TemporalData,
)
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


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 32,
    num_ctx_min: int = 32,
    num_ctx_max: int = 256,
    num_test: int = 256,
):
    (f_train, X_train), (f_valid, X_valid), (f_test, X_test) = load_data(rng)

    def build_dataloader(f, X):
        N, L = X.shape[0], num_ctx_max + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_b, rng_i, rng = random.split(rng, 3)
                rng_bs = random.split(rng_b, batch_size)
                idx = vmap(lambda rng: random.choice(rng, N, (L,), replace=False))(
                    rng_bs
                )  # [B, L]
                yield TemporalData(x=X[idx], f=f[idx]).batch(
                    rng_i,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                )

        return dataloader

    # NOTE: uncomment to use _entire_ valid set, similar to test set (much slower)
    # def valid_dataloader(rng: jax.Array):
    #     yield TabularBatch(
    #         x_ctx=X_train[None, ...],
    #         f_ctx=f_train[None, ...],
    #         mask_ctx=jnp.ones((1, X_train.shape[0]), dtype=bool),
    #         x_test=X_valid[None, ...],
    #         f_test=f_valid[None, ...],
    #     )

    def test_dataloader(rng: jax.Array):
        N = X_train.shape[0] + X_valid.shape[0]
        yield TabularBatch(
            x_ctx=jnp.concat([X_train, X_valid], axis=0)[None, ...],
            f_ctx=jnp.concat([f_train, f_valid], axis=0)[None, ...],
            mask_ctx=jnp.ones((1, N), dtype=bool),
            x_test=X_test[None, ...],
            f_test=f_test[None, ...],
        )

    return (
        build_dataloader(f_train, X_train),
        build_dataloader(f_valid, X_valid),
        test_dataloader,
    )


def load_data(rng: jax.Array):
    try:
        df = pd.read_csv("cache/household_power_consumption.txt", sep=";").dropna()
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
    df["dt"] = pd.to_datetime(df.Date + " " + df.Time, dayfirst=True)
    df["year"] = df.dt.dt.year
    df["min_of_year"] = df.dt.dt.day_of_year * 24 * 60
    df = df.drop(columns=["Date", "Time", "dt"])
    fX = df.values
    N = fX.shape[0]
    pct_train, pct_test = 0.8, 0.1
    num_train, num_test = int(pct_train * N), int(pct_test * N)
    perm = random.permutation(rng, N)
    fX = fX[perm]
    fX_train, fX_valid, fX_test = (
        fX[:num_train],
        fX[num_train:-num_test],
        fX[-num_test:],
    )
    scaler = StandardScaler()
    fX_train = scaler.fit_transform(fX_train)
    fX_valid = scaler.transform(fX_valid)
    fX_test = scaler.transform(fX_test)
    f_train, X_train = fX_train[:, [0]], fX_train[:, 1:]
    f_valid, X_valid = fX_valid[:, [0]], fX_valid[:, 1:]
    f_test, X_test = fX_test[:, [0]], fX_test[:, 1:]
    return (f_train, X_train), (f_valid, X_valid), (f_test, X_test)


if __name__ == "__main__":
    main()
