#!/usr/bin/env python3
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random


class Conditional_MLP(nn.Module):
    mlp_model: nn.Module

    @nn.compact
    def __call__(self, x: Array, conditionals: list[Array], **kwargs):
        B = x.shape[0]
        batched_conditionals = jnp.repeat(
            jnp.stack(conditionals).reshape(1, -1), repeats=B, axis=0
        )
        x = jnp.hstack([x.reshape(B, -1), batched_conditionals])
        return self.mlp_model(x)


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

    encoder: Optional[nn.Module]
    decoder: nn.Module
    z_dim: int
    decoder_only: bool = False

    @nn.compact
    def __call__(
        self, f: Array, conditionals: list[Array], decode_only: bool = False, **kwargs
    ):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, K, 1)`.
            conditionals: list of conditional variavles to condition the
            output on. NOTE: Must ensure same ordering of conditional
            at all times, as the model implicitly learns this order.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        if decode_only or self.decoder_only:
            z, z_mu, z_std = f, None, None
        else:
            latents = self.encoder(f, conditionals)
            z_mu = nn.Dense(self.z_dim)(latents)
            z_log_var = nn.Dense(self.z_dim)(latents)
            z_std = jnp.exp(z_log_var / 2)
            eps = random.normal(self.make_rng("extra"), z_std.shape)
            z = z_mu + z_std * eps
        f_hat = self.decoder(z, conditionals)
        return f_hat.reshape(f.shape), z_mu, z_std
