#!/usr/bin/env python3
from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random


@dataclass
class PiVAE(nn.Module):
    r"""PiVAE approximates a stochastic process.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate samples from the approximated process.

    Args:
        encoder: A module used to encode GP realizations and
            their hyperparamters.
        decoder: A module used to decode random vectors and
            GP hyperparameters into GP samples.
        beta_dim: The size of the hidden dimension.

    Returns:
        TODO(danj): Finish
    """

    encoder: nn.Module
    decoder: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, rng: Array, f: Array):
        batch_size = f.shape[0]
        latents = self.encoder(f.reshape(batch_size, -1))
        mu = nn.Dense(self.z_dim)(latents)
        log_var = nn.Dense(self.z_dim)(latents)
        std = jnp.exp(log_var / 2)
        eps = random.normal(rng, log_var.shape)
        z = mu + std * eps
        f_hat = self.decoder(z)
        return f_hat.reshape(f.shape), mu, log_var
