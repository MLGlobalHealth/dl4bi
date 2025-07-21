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
from sps.kernels import matern_3_2
from sps.utils import build_grid

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import (
    MLPDeepRV,
    PriorCVAE,
    ScanTransformerDeepRV,
    gMLPDeepRV,
)
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    prior_cvae_train_step,
)


def main(seed=15):
    save_dir = Path("results/optimization_test/")
    save_dir.mkdir(parents=True, exist_ok=True)
    grids = [
        build_grid([{"start": 0.0, "stop": 100.0, "num": n}])
        for n in [256, 512, 1024, 2048, 4096]
    ]
    priors = {"ls": dist.Uniform(0.0, 100.0)}
    result = []
    rng = random.key(seed)
    for s in grids:
        rng, _ = random.split(rng)
        L = s.shape[0]
        (save_dir / f"grid_{L}").mkdir(parents=True, exist_ok=True)
        models = {
            "PriorCVAE": PriorCVAE(MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L),
            "DeepRV + MLP": MLPDeepRV(dims=[L, L]),
            "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
            "DeepRV + gMLP Large": gMLPDeepRV(num_blks=4),
            "DeepRV + ScanTransfomer": ScanTransformerDeepRV(num_blks=2, dim=64),
            "DeepRV + ScanTransfomer Large": ScanTransformerDeepRV(num_blks=4, dim=64),
        }
        default_steps = 100_000 if L <= 1024 else 200_000
        rng_train, rng_test = random.split(rng, 2)
        for model_name, nn_model in models.items():
            optimizer, max_lr, bs, train_steps = gen_train_params(
                model_name, s, default_steps
            )
            loader = gen_gp_dataloader(s, priors, matern_3_2, batch_size=bs)
            wandb.init(
                config={
                    "model_name": model_name,
                    "grid_size": L,
                    "max_lr": max_lr,
                    "batch_size": bs,
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
    max_lr = {
        "PriorCVAE": 1e-3,
        "DeepRV + MLP": 5e-3,
        "DeepRV + gMLP": 5e-3,
        "DeepRV + gMLP Large": 5e-3,
        "DeepRV + ScanTransfomer": 1e-4,
        "DeepRV + ScanTransfomer Large": 1e-4,
    }[model_name]
    bs = int(min(1, 512 / L) * default_bs)
    train_steps = default_steps * (default_bs // bs)
    optimizer = optax.yogi(cosine_annealing_lr(train_steps, max_lr))
    if model_name in [
        "DeepRV + ScanTransfomer",
        "DeepRV + ScanTransfomer Large",
    ]:
        optimizer = optax.adamw(max_lr)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
    return optimizer, max_lr, bs, train_steps


if __name__ == "__main__":
    main()
