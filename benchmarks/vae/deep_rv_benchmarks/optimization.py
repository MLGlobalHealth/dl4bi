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
    DKADeepRV,
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
    default_bs = 32
    default_steps = 100_000
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
        for max_lr in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            models = {
                "PriorCVAE": PriorCVAE(
                    MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L
                ),
                "DeepRV + MLP": MLPDeepRV(dims=[L, L]),
                "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
                "DeepRV + ScanTransfomer": ScanTransformerDeepRV(num_blks=2, dim=64),
                "DeepRV + DKA": DKADeepRV(num_blks=2, dim=64),
            }
            rng_train, rng_test = random.split(rng, 2)
            for model_name, nn_model in models.items():
                scan_bs = int(min(1, 512 / L) * default_bs)
                bs = scan_bs if model_name == "DeepRV + ScanTransfomer" else default_bs
                train_steps = default_steps * (default_bs // bs)
                loader = gen_gp_dataloader(s, priors, matern_3_2, batch_size=bs)
                optimizer = optax.yogi(
                    cosine_annealing_lr(train_steps, max_lr, lr_min=0.0)
                )
                if model_name in ["DeepRV + ScanTransfomer", "DeepRV + DKA"]:
                    optimizer = optax.adamw(max_lr)
                optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
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
        early_stop_patience=2,
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


if __name__ == "__main__":
    main()
