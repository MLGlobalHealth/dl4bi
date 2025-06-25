#!/usr/bin/env python3
"""
This script gives an example of Noise Contrastive Estimation (NCE).
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training.train_state import TrainState
from jax import jit, random
from jax.nn import sigmoid
from jax.scipy import stats
from tqdm import tqdm

from dl4bi.core.mlp import MLP


def main():
    num_steps, batch_size = 1000, 512
    rng = random.key(42)
    optimizer = optax.adamw(1e-3)
    x = sample_gmm(rng, batch_size)
    model = MLP([128, 128, 128], nn.gelu)
    params = model.init(rng, x)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    pbar = tqdm(range(num_steps), unit="batches")
    for i in pbar:
        rng_i, rng = random.split(rng)
        state, loss = train_step(rng_i, state, batch_size)
        if i % 100 == 0:
            pbar.set_postfix({"loss": loss.item()})
    x = jnp.linspace(-20, 20, 10000).reshape(-1, 1)
    log_pm = state.apply_fn({"params": state.params}, x)
    plt.plot(np.array(x).flatten(), np.array(gmm_density(x)).flatten())
    plt.plot(np.array(x).flatten(), np.exp(log_pm).flatten())
    plt.savefig("nce.png")


@partial(jit, static_argnames=("n",))
def sample_gmm(rng: jax.Array, n: int):
    rng_i, rng_z = random.split(rng)
    w = jnp.array([0.2, 0.7, 0.1])
    mu = jnp.array([-5.0, 0.0, 5.0])
    sigma = jnp.array([2.0, 0.5, 3.0])
    idx = random.choice(rng_i, 3, (n,), p=w)
    z = random.normal(rng_z)
    return (mu[idx] + sigma[idx] * z).reshape(-1, 1)


def gmm_density(x: jax.Array):
    return (
        0.2 * stats.norm.pdf(x, -5.0, 2.0)
        + 0.7 * stats.norm.pdf(x, 0.0, 0.5)
        + 0.1 * stats.norm.pdf(x, 5.0, 3.0)
    )


@partial(jit, static_argnames=("n",))
def sample_noise(rng: jax.Array, n: int):
    return (0 + 5.0 * random.normal(rng, (n,))).reshape(n, 1)


def log_pn(x: jax.Array):
    return stats.norm.logpdf(x, 0, 5.0)


@partial(jit, static_argnames=("batch_size",))
def train_step(rng: jax.Array, state: TrainState, batch_size: int):
    rng_d, rng_n = random.split(rng)
    data = sample_gmm(rng_d, batch_size)
    noise = sample_noise(rng_n, batch_size)

    def nce_loss(params):
        log_pm_data = state.apply_fn({"params": params}, data)
        log_pn_data = log_pn(data)
        loss_data = sigmoid(log_pm_data - log_pn_data)
        log_pm_noise = state.apply_fn({"params": params}, noise)
        log_pn_noise = log_pn(noise)
        loss_noise = 1 - sigmoid(log_pm_noise - log_pn_noise)
        return -(jnp.log(loss_data).mean() + jnp.log(loss_noise).mean()) / 2

    loss, grads = jax.value_and_grad(nce_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


if __name__ == "__main__":
    main()
