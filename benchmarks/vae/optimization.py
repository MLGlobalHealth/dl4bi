import sys

sys.path.append("benchmarks/vae")
from datetime import datetime
from pathlib import Path
from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from sps.kernels import matern_1_2, matern_3_2, rbf
from sps.utils import build_grid

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    prior_cvae_train_step,
)


def main(seed=15):
    save_dir = Path("results/optimization_test/")
    save_dir.mkdir(parents=True, exist_ok=True)
    grids = [
        build_grid([{"start": 0.0, "stop": 100.0, "num": n}] * 2).reshape(-1, 2)
        for n in [16, 32, 64]
    ]
    priors = {"ls": dist.Uniform(1.0, 100.0)}
    result = []
    rng = random.key(seed)
    for s in grids:
        rng, _ = random.split(rng)
        L = s.shape[0]
        (save_dir / f"grid_{L}").mkdir(parents=True, exist_ok=True)
        models = {
            f"DeepRV + gMLP AdamW clip={clip_k} decay={decay} lr={lr}": gMLPDeepRV(
                num_blks=2
            )
            for clip_k in ["0.5", "1.0", "3.0"]
            for decay in [1e-3, 1e-2]
            for lr in ([1e-3] if L <= 48**2 else [2e-3])
        }
        default_steps = 200_000 if L <= 1024 else 300_000
        rng_train, rng_test = random.split(rng, 2)
        for kernel in [rbf, matern_3_2, matern_1_2]:
            for model_name, nn_model in models.items():
                optimizer, max_lr, bs, train_steps = gen_train_params(
                    model_name, s, default_steps
                )
                loader = gen_gp_dataloader(s, priors, kernel, batch_size=bs)
                wandb.init(
                    config={
                        "model_name": model_name,
                        "grid_size": L,
                        "max_lr": max_lr,
                        "batch_size": bs,
                        "kernel": kernel.__name__,
                    },
                    mode="online",
                    name=f"{model_name}",
                    project="deep_rv_optimizations",
                    reinit=True,
                )
                train_time, eval_mse, _ = surrogate_model_train(
                    rng_train,
                    rng_test,
                    loader,
                    nn_model,
                    optimizer,
                    train_steps,
                )
                result.append(
                    {
                        "model_name": model_name,
                        "train_time": train_time,
                        "Test Norm MSE": eval_mse,
                        "max_lr": max_lr,
                        "grid_size": L,
                        "batch_size": bs,
                        "kernel": kernel.__name__,
                    }
                )
                wandb.log(
                    {
                        "train_time": train_time,
                        "Test Norm MSE": eval_mse,
                    }
                )
        pd.DataFrame(result).to_csv((save_dir / f"grid_{L}") / "res.csv")


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    optimizer,
    train_num_steps: int,
    valid_interval: int = 25_000,
    valid_steps: int = 5_000,
):
    train_step = prior_cvae_train_step
    if model.__class__.__name__ != "PriorCVAE":
        train_step = deep_rv_train_step
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        train_num_steps,
        loader,
        valid_step,
        valid_interval,
        valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, valid_steps)["norm MSE"]
    return train_time, eval_mse, state


def gen_gp_dataloader(s: Array, priors: dict, kernel: Callable, batch_size=32):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_train_params(model_name, s, default_steps, default_bs=32):
    L = s.shape[0]
    max_lr = float(model_name.split("lr=")[-1])
    bs = default_bs if L <= 48**2 else default_bs // 2
    train_steps = default_steps * (default_bs // bs)
    clip_vals = {
        "0.5": optax.clip_by_global_norm(0.5),
        "1.0": optax.clip_by_global_norm(1.0),
        "3.0": optax.clip_by_global_norm(3.0),
    }
    optimizer = {
        f"DeepRV + gMLP AdamW clip={clip_k} decay={decay} lr={lr}": optax.chain(
            clip_vals[clip_k],
            optax.adamw(
                learning_rate=cosine_annealing_lr(train_steps, lr), weight_decay=decay
            ),
        )
        for clip_k in ["0.5", "1.0", "3.0"]
        for decay in [1e-3, 1e-2]
        for lr in ([1e-3] if L <= 48**2 else [2e-3])
    }[model_name]
    return optimizer, max_lr, bs, train_steps


if __name__ == "__main__":
    main()
