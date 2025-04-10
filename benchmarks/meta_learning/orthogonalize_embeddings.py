#!/usr/bin/env python3
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random

from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import train


def main():
    wandb.init(mode="disabled")
    H, D_x = 512, 768
    model = VAE(
        encoder=MLP([H, H, 2 * H], nn.gelu),
        decoder=MLP([H, H, D_x], nn.gelu),
    )
    dataloader = build_dataloader()
    state = train(
        rng=random.key(42),
        model=model,
        optimizer=optax.yogi(1e-4),
        train_step=model.train_step,
        train_num_steps=100000,
        train_dataloader=dataloader,
        valid_step=model.valid_step,
        valid_interval=10000,
        valid_num_steps=1000,
        valid_dataloader=dataloader,
        valid_monitor_metric="MSE",
    )
    x = np.load("cache/skin_lesions/train_x.npy", allow_pickle=True)
    z, z_mu, s_std = state.apply_fn(
        {"params": state.params},
        x=x,
        rngs={"extra": random.key(42)},
        method="encode",
    )
    jnp.save("cache/skin_lesions/train_x_ortho.npy", z)


def build_dataloader(batch_size: int = 32):
    x = np.load("cache/skin_lesions/train_x.npy", mmap_mode="r", allow_pickle=True)
    N = x.shape[0]

    def dataloader(rng: jax.Array):
        while True:
            rng_idx, rng = random.split(rng)
            idx = random.choice(rng_idx, N, (batch_size,))
            yield {"x": x[idx]}

    return dataloader


@jit
def train_step(rng, state, batch):
    def loss_fn(params):
        output = state.apply_fn({"params": params}, **batch, rngs={"extra": rng})
        return output.mse(batch["x"]) + output.kl_normal_dist()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def valid_step(rng: jax.Array, state: TrainState, batch: Mapping):
    output = state.apply_fn({"params": state.params}, **batch, rngs={"extra": rng})
    return output.metrics(batch["x"])


class VAE(nn.Module):
    encoder: Callable
    decoder: Callable
    train_step: Callable = train_step
    valid_step: Callable = valid_step
    output_fn: Callable = VAEOutput.from_raw_output

    @nn.compact
    def __call__(self, x: jax.Array):  # x: [B, D_x]
        z, z_mu, z_std = self.encode(x)
        x_hat = self.decode(z)
        return self.output_fn(x_hat, z_mu, z_std)

    def encode(self, x: jax.Array):  # [B, D_x]
        latents = self.encoder(x)
        z_mu, z_log_var = jnp.split(latents, 2, axis=-1)
        z_std = jnp.exp(0.5 * z_log_var)
        eps = random.normal(self.make_rng("extra"), z_mu.shape)
        z = z_mu + z_std * eps
        return z, z_mu, z_std

    def decode(self, z: jax.Array):  # [B, D_z]
        return self.decoder(z)


if __name__ == "__main__":
    main()
