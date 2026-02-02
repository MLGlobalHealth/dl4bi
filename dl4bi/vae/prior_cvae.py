#!/usr/bin/env python3
from collections.abc import Callable

from flax import linen as nn
from jax import Array, random

from dl4bi.core.model_output import VAEOutput


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
    cond_stack_fn: Callable
    z_dim: int

    @nn.compact
    def __call__(self, f: Array, conditionals: Array, **kwargs):
        r"""Run module forward.

        Args:
            f: The function values, an array of shape `(B, K, 1)`.
            conditionals: The conditional hyperparameters of the stochastic process

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        latents = self.encoder(self.cond_stack_fn(f, conditionals), **kwargs).squeeze()
        z_mu = nn.Dense(self.z_dim)(latents)
        z_log_var = nn.Dense(self.z_dim)(latents)
        z_std = nn.softplus(z_log_var / 2)
        eps = random.normal(self.make_rng("extra"), z_std.shape)
        z = z_mu + z_std * eps
        f_hat = self.decoder(self.cond_stack_fn(z, conditionals), **kwargs)
        return VAEOutput.from_raw_output(f_hat.reshape(f.shape), z_mu, z_std)

    def decode(self, z: Array, conditionals: Array, **kwargs):
        return self.decoder(self.cond_stack_fn(z, conditionals), **kwargs)
