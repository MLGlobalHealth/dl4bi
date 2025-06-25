#!/usr/bin/env python3
"""
This script gives an example of contrastive predictive coding (CPC).
"""

from functools import partial
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import jit, random
from tqdm import tqdm

from dl4bi.core.mlp import MLP


def main():
    rng = random.key(42)
    optimizer = optax.adamw(1e-3)
    model = CPCModel()
    x = sample_batch(rng)
    params = model.init(rng, x)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    pbar = tqdm(range(1000), unit="batches")
    for i in pbar:
        rng_i, rng = random.split(rng)
        state, loss = train_step(rng_i, state)
        if i % 100 == 0:
            pbar.set_postfix({"loss": loss.item()})


@partial(jit, static_argnames=("shape",))
def sample_batch(rng: jax.Array, shape: tuple[int, int, int] = (64, 128, 4)):
    B, L, D = shape
    x = random.normal(rng, (B, L, D))
    # simple autoregressive process
    for i in range(1, L):
        x = x.at[:, i, :].set(x[:, i, :] + 0.8 * x[:, i - 1, :])
    return x


class GRU(nn.Module):
    dim: int = 64

    @nn.compact
    def __call__(self, x):
        B, D = x.shape[0], self.dim
        cell = nn.scan(
            nn.GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )(name="cell", features=D)
        carry = self.param("carry", nn.initializers.zeros, (1, D))
        carry = jnp.tile(carry, (B, 1))
        carry, y = cell(carry, x)
        return y  # [B, T, D]


class CPCModel(nn.Module):
    encode: Callable = MLP([64] * 4)
    autoregress: Callable = GRU(64)

    @nn.compact
    def __call__(self, x):
        z = self.encode(x)
        c = self.autoregress(z)
        return z, c


@partial(jit, static_argnames=("K",))
def train_step(rng: jax.Array, state: TrainState, K: int = 5):
    x = sample_batch(rng)

    def loss_fn(params):
        z, c = state.apply_fn({"params": params}, x)
        B, T, D = z.shape
        z_k = jnp.stack([jnp.roll(z, -k - 1, axis=1) for k in range(K)], axis=2)
        logits = jnp.einsum("btd,btkd->btk", c, z_k)
        log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
        # positive is z_{t+1}, negatives are z_{t+2:t+k}
        # since cross-entropy would zero those out, we only take the mean of the
        # first prediction for each time step
        return -log_probs[:, T - K, 0].mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


if __name__ == "__main__":
    main()
