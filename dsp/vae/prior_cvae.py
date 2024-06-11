#!/usr/bin/env python3
import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random


class PriorCVAE(nn.Module):
    r"""[PriorCVAE](https://arxiv.org/pdf/2304.04307) approximates a Gaussian Process.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate a GP from the samples it was trained on.

    Args:
        encoder: A module used to encode GP realizations and
            their hyperparamters.
        decoder: A module used to decode random vectors and
            GP hyperparameters into GP samples.
        z_dim: The size of the hidden dimension.

    Returns:
        An instance of the PriorCVAE network.
        $\hat{\mathbf{f}}$, a recreation of the original $\mathbf{f}$,
        along with $\mu$ and $\log(\sigma^2)$, which are often used
        to calculate losses involving KL divergence.
    """

    encoder: nn.Module
    decoder: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, rng: Array, var: Array, ls: Array, f: Array):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            var: The variance for the GP.
            ls: The lengthscale for the GP.
            f: The function values, an array of shape `(B, K, 1)`.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        if f.shape != var.shape:  # scalars
            batch_size = f.shape[0]
            var = jnp.full((batch_size, 1), var)
            ls = jnp.full((batch_size, 1), ls)
        z_mu, z_std = self.encode(var, ls, f)
        eps = random.normal(rng, z_std.shape)
        z = z_mu + z_std * eps
        f_hat = self.decode(z, var, ls)
        return f_hat.reshape(f.shape), z_mu, z_std

    def encode(self, var: Array, ls: Array, f: Array):
        f_flat = f.reshape(f.shape[0], -1)
        latents = self.encoder(jnp.hstack([f_flat, var, ls]))
        z_mu = nn.Dense(self.z_dim)(latents)
        z_log_var = nn.Dense(self.z_dim)(latents)
        z_std = jnp.exp(z_log_var / 2)
        return z_mu, z_std

    def decode(self, z: Array, var: Array, ls: Array):
        return self.decoder(jnp.hstack([z, var, ls]))
