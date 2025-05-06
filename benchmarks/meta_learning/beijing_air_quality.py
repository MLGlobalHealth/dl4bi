#!/usr/bin/env python3
import sys
from io import StringIO
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from hydra.utils import instantiate
from jax import random, vmap
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dl4bi.core.train import (
    Callback,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import (
    SpatiotemporalBatch,
    SpatiotemporalData,
)
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
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


# TODO(danj): TaublarData needs to support space and time
def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 16,
    num_ctx_min: int = 16,
    num_ctx_max: int = 128,
    num_test: int = 256,
):
    train, valid, test = load_data(rng)
    x_train, s_train, t_train, f_train = train
    x_valid, s_valid, t_valid, f_valid = valid
    x_test, s_valid, t_test, f_test = test

    def build_dataloader(x, s, t, f):
        N, L = X.shape[0], num_ctx_max + num_test

        def dataloader(rng: jax.Array):
            while True:
                rng_b, rng_i, rng = random.split(rng, 3)
                rng_bs = random.split(rng_b, batch_size)
                idx = vmap(lambda rng: random.choice(rng, N, (L,), replace=False))(
                    rng_bs
                )  # [B, L]
                yield SpatiotemporalData(x[idx], s[idx], t[idx], f[idx]).batch(
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
        build_dataloader(f_train, X_train),
        build_dataloader(f_valid, X_valid),
        test_dataloader,
    )


def load_data(rng: jax.Array):
    dir = Path("cache/beijing_air_quality")
    try:
        df = pd.concat([pd.read_csv(p) for p in dir.glob("*.csv")], ignore_index=True)
        df = df.dropna()
        df_coords = load_coords()
        df = df.merge(df_coords, on="station", how="left")
        df = df.drop(columns=["No", "station"])
    except FileNotFoundError:
        url = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
        msg = f"""
        1. Download the dataset here: {url}
        2. Unzip the dataset
        3. Move files from the PRSA_* directory into "{dir}"
        """
        print(msg)
        sys.exit("Dataset not available.")
    N = df.shape[0]
    N_train = int(N * 0.8)
    N_test = int(N * 0.1)
    permute_idx = random.choice(rng, N, (N,), replace=False)
    df = df.iloc[permute_idx]
    df_train, df_valid, df_test = df[:N_train], df[N_train:-N_test], df[-N_test:]
    df_train, df_valid, df_test = standardize_by_train(df_train, df_valid, df_test)


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
):
    num_feats = list(set(df_train.columns) - {"wd"})
    cat_feats = ["wd"]
    tx = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(sparse_output=False), cat_feats),
        ],
        remainder="passthrough",
    )
    x_train = tx.fit_transform(df_train)
    x_valid = tx.transform(df_valid)
    x_test = tx.transform(df_test)
    tx_onehot = tx.named_transformers_["cat"]
    cat_feats = tx_onehot.get_feature_names_out(cat_feats).tolist()
    cols = num_feats + cat_feats
    df_train = pd.DataFrame(x_train, columns=cols)
    df_valid = pd.DataFrame(x_valid, columns=cols)
    df_test = pd.DataFrame(x_test, columns=cols)
    return df_train, df_valid, df_test


if __name__ == "__main__":
    main()
