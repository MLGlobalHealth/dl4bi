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
    def __call__(self, f: Array, var: float, ls: float):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, K, 1)`.
            var: The variance for the GP.
            ls: The lengthscale for the GP.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        B = f.shape[0]
        var = jnp.full((B, 1), var)
        ls = jnp.full((B, 1), ls)
        latents = self.encoder(jnp.hstack([f.reshape(B, -1), var, ls]))
        z_mu = nn.Dense(self.z_dim)(latents)
        z_log_var = nn.Dense(self.z_dim)(latents)
        z_std = jnp.exp(z_log_var / 2)
        eps = random.normal(self.make_rng("extra"), z_std.shape)
        z = z_mu + z_std * eps
        f_hat = self.decoder(jnp.hstack([z, var, ls]))
        return f_hat.reshape(f.shape), z_mu, z_std

    def decode(self, z: Array, var: float, ls: float):
        B = z.shape[0]
        var = jnp.full((B, 1), var)
        ls = jnp.full((B, 1), ls)
        return self.decoder(jnp.hstack([z, var, ls]))
