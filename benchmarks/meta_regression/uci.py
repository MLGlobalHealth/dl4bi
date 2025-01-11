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
from ucimlrepo import fetch_ucirepo

from dl4bi.meta_regression.train_utils import (
    cfg_to_run_name,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    save_ckpt,
    train,
)


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
        cfg.data.name,
        cfg.data.batch_size,
        cfg.data.num_ctx.min,
        cfg.data.num_ctx.max,
        cfg.data.num_test,
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


def build_dataloaders(
    rng: jax.Array,
    name: str,
    batch_size: int,
    num_ctx_min: int,
    num_ctx_max: int,
    num_test: int,
):
    df, target = load_dataset(name)
    features = list(set(df.columns) - {target})
    N, B = df.shape[0], batch_size
    N_train, N_test = int(N * 4 / 9), int(N * 3 / 9)  # same as 1M GP paper
    permute_idx = random.choice(rng, N, (N,), replace=False)
    # TODO(danj): perform whitening
    df = df.iloc[permute_idx]
    df_train = df[:N_train]
    df_valid = df[N_train:-N_test]
    df_test = df[-N_test:]

    def build_dataloader(df):
        L = df.shape[0]
        valid_lens_test = jnp.repeat(num_ctx_max + num_test, B)

        def dataloader(rng: jax.Array):
            while True:
                rng_permute, rng_valid, rng = random.split(rng, 3)
                permute_idx = random.choice(rng_permute, L, (L,), replace=False)
                dft = df.iloc[permute_idx][: num_ctx_max + num_test]
                s_test = jnp.array(dft[features].values)[None, ...]
                f_test = jnp.array(dft[target].values)[None, :, None]
                s_test = jnp.repeat(s_test, B, axis=0)
                f_test = jnp.repeat(f_test, B, axis=0)
                # permute the order and select the first valid_lens_ctx for context
                valid_lens_ctx = random.randint(
                    rng_valid,
                    (B,),
                    num_ctx_min,
                    num_ctx_max,
                )
                yield (
                    s_test,
                    f_test,
                    valid_lens_ctx,
                    s_test,
                    f_test,
                    valid_lens_test,
                )

        return dataloader

    return (
        build_dataloader(df_train),
        build_dataloader(df_valid),
        build_dataloader(df_test),
    )


def load_dataset(name: str):
    target = {"bike": "count"}[name]
    path = Path(f"cache/uci/{name}.csv")
    if path.exists():
        df = pd.read_csv(path)
        return df, target
    path.parent.mkdir(parents=True, exist_ok=True)
    match name:
        case "bike":
            data = fetch_ucirepo(id=275)
            df = data.data.features.drop("dteday", axis=1)
            df["count"] = data.data.targets["cnt"]
        case _:
            raise Exception(f"Invalid dataset {name}!")
    df.to_csv(path, index=False)
    return df, target


if __name__ == "__main__":
    main()
