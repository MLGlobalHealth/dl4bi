import sys

sys.path.append("benchmarks/vae")
from datetime import datetime
from pathlib import Path
from typing import Callable, Union

import flax.linen as nn
import jax.numpy as jnp
import optax
import pandas as pd
from jax import Array, jit, random, vmap
from numpyro import distributions as dist
from sps.kernels import matern_1_2, matern_5_2, rbf

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import FixedKernelAttention, gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step


def main(seed=15):
    save_dir = Path("results/multi_location/")
    save_dir.mkdir(parents=True, exist_ok=True)
    grid_sizes = [256]  # [256, 1024]
    max_s = 100.0
    priors = {"ls": dist.Uniform(1.0, max_s)}
    rng = random.key(seed)
    for L in grid_sizes:
        result = []
        grid_dir = save_dir / f"grid_{L}"
        grid_dir.mkdir(parents=True, exist_ok=True)
        default_steps = 600_000
        optimizer, max_lr, bs, train_steps = gen_train_params(L, default_steps)

        for kernel in [matern_1_2, matern_5_2, rbf]:
            rng, _ = random.split(rng)
            models = {
                "DeepRV + gMLP large": gMLPDeepRV(num_blks=8, head=MLP([128, 64, 1])),
                "DeepRV + gMLP kernel attn large": gMLPDeepRV(
                    num_blks=8, attn=FixedKernelAttention(), head=MLP([128, 64, 1])
                ),
                "DeepRV + gMLP Fourier 14 large": gMLPDeepRV(
                    num_blks=8,
                    s_embed=FourierEmbed(num_bands=14),
                    head=MLP([128, 64, 1]),
                ),
                "DeepRV + gMLP RFF 512 large": gMLPDeepRV(
                    num_blks=8,
                    s_embed=RFFEmbed(num_features=512),
                    head=MLP([128, 64, 1]),
                ),
            }
            rng_train, rng_test = random.split(rng, 2)
            multi_s_loader = multi_s_dataloader(L, priors, kernel, max_s, batch_size=bs)
            for model_name, nn_model in models.items():
                single_s_loader = single_s_dataloader(
                    model_name,
                    L,
                    priors,
                    kernel,
                    max_s,
                    batch_size=bs,
                )
                wandb.init(
                    config={
                        "model_name": model_name,
                        "grid_size": L,
                        "max_lr": max_lr,
                        "batch_size": bs,
                        "kernel": kernel.__name__,
                    },
                    mode="online",
                    name=f"{model_name}_{kernel.__name__}",
                    project="deep_rv_multiple_s",
                    reinit=True,
                )
                train_time, eval_mse, _ = surrogate_model_train(
                    rng_train,
                    rng_test,
                    multi_s_loader if "batched" in model_name else single_s_loader,
                    single_s_loader,
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
                wandb.log({"train_time": train_time, "Test Norm MSE": eval_mse})
        pd.DataFrame(result).to_csv(grid_dir / "res.csv")


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    train_loader: Callable,
    valid_loader: Callable,
    model: nn.Module,
    optimizer,
    train_num_steps: int,
    valid_interval: int = 100_000,
    valid_steps: int = 5_000,
):
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        deep_rv_train_step,
        train_num_steps,
        train_loader,
        valid_step,
        valid_interval,
        valid_steps,
        valid_loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, valid_loader, valid_steps)[
        "norm MSE"
    ]
    return train_time, eval_mse, state


def single_s_dataloader(
    model_name: str,
    grid_size: int,
    priors: dict,
    kernel: Callable,
    max_s: float,
    batch_size=32,
):
    jitter = 5e-4 * jnp.eye(grid_size)
    s_jit = jit(lambda rng: random.uniform(rng, (grid_size, 2), maxval=max_s))
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))
    s_batch_jit = jit(lambda s: s)
    cond_batch_jit = jit(lambda ls: jnp.array([ls]))
    z_mixed_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", K, z))
    if "batched" in model_name:
        s_batch_jit = jit(lambda s: jnp.repeat(s[None], batch_size, axis=0))
        cond_batch_jit = jit(lambda ls: jnp.full((batch_size, grid_size, 1), ls))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z, rng_samp = random.split(rng_data, 4)
            var = 1.0
            s = s_jit(rng_samp)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, grid_size))
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z)
            yield {
                "s": s_batch_jit(s),
                "f": f,
                "z": z,
                "conditionals": cond_batch_jit(ls),
                "mixed_z": z_mixed_jit(K, z),
                "K": K,
            }

    return dataloader


def multi_s_dataloader(
    grid_size: int,
    priors: dict,
    kernel: Callable,
    max_s: float = 100.0,
    batch_size=32,
):
    jitter = 5e-4 * jnp.eye(grid_size)[None,]
    s_jit = jit(
        lambda rng: random.uniform(rng, (batch_size, grid_size, 2), maxval=max_s)
    )
    kernel_jit = jit(
        vmap(lambda s, var, ls: kernel(s, s, var, ls), in_axes=(0, None, 0))
    )
    f_jit = jit(lambda K, z: jnp.einsum("bij,bj->bi", jnp.linalg.cholesky(K), z))
    batch_ls = jit(lambda ls: jnp.repeat(ls[:, None], grid_size, axis=1))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z, rng_samp = random.split(rng_data, 4)
            var = 1.0
            s = s_jit(rng_samp)
            ls = priors["ls"].sample(rng_ls, sample_shape=(batch_size, 1))
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, grid_size))
            K = kernel_jit(s, var, ls) + jitter
            f = f_jit(K, z)
            yield {"s": s, "f": f, "z": z, "conditionals": batch_ls(ls), "K": K}

    return dataloader


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_train_params(grid_size, default_steps, default_bs=32):
    L = grid_size
    bs = int(min(1, 2048 / L) * default_bs)
    train_steps = default_steps * (default_bs // bs)
    max_lr = 5e-3 if L <= 1024 else 1e-2
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.yogi(cosine_annealing_lr(train_steps, max_lr)),
    )
    return optimizer, max_lr, bs, train_steps


class FourierEmbed(nn.Module):
    num_bands: int = 6
    include_original: bool = True
    head: Union[Callable, nn.Module] = lambda x: x

    @nn.compact
    def __call__(self, s: Array):
        # Fourier encode s: (L, D) -> (L, D_encoded)
        freq_bands = 2.0 ** jnp.arange(self.num_bands)
        s_expanded = s[:, :, None] * freq_bands[None, None, :]
        s_expanded = jnp.pi * s_expanded
        sin = jnp.sin(s_expanded)
        cos = jnp.cos(s_expanded)
        s_encoded = jnp.concatenate([sin, cos], axis=-1).reshape(s.shape[0], -1)
        if self.include_original:
            s_encoded = jnp.concatenate([s, s_encoded], axis=-1)
        return self.head(s_encoded)


class RFFEmbed(nn.Module):
    num_features: int = 128
    scale: float = 1.0
    include_original: bool = True
    head: Union[Callable, nn.Module] = lambda x: x

    @nn.compact
    def __call__(self, s: Array):
        D = s.shape[-1]
        W = self.param(
            "rff_W",
            lambda k: random.normal(k, shape=(self.num_features, D)) * self.scale,
        )
        b = self.param(
            "rff_b",
            lambda k: random.uniform(
                k, shape=(self.num_features,), minval=0.0, maxval=2 * jnp.pi
            ),
        )
        s_proj = jnp.dot(s, W.T) + b
        rff = jnp.sqrt(2.0 / self.num_features) * jnp.cos(s_proj)
        if self.include_original:
            s_encoded = jnp.concatenate([s, rff], axis=-1)
        else:
            s_encoded = rff
        return self.head(s_encoded)


# TODO(jhoott): move to batched training if it shows improvement, else remove
# class gMLPActivBatchedDeepRV(nn.Module):
#     num_blks: int = 2

#     @nn.compact
#     def __call__(self, z: Array, conditionals: Array, s: Array, **kwargs):
#         x = jnp.concat([jnp.atleast_3d(z), s, conditionals], axis=-1)
#         return VAEOutput(
#             gMLP(
#                 num_blks=self.num_blks,
#                 embed=MLP([64, 64], nn.gelu),
#                 blk=gMLPBlock(
#                     proj_in=MLP([128, 128], nn.gelu),
#                     proj_out=MLP([64, 64], nn.gelu),
#                 ),
#             )(x)
#         )

#     def decode(self, z: Array, conditionals: Array, s: Array, **kwargs):
#         return self(z, conditionals, s, **kwargs).f_hat


if __name__ == "__main__":
    main()
