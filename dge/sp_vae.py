#!/usr/bin/env python3
import jax.numpy as jnp
from flax import linen as nn
from jax import Array, random


class SPVAE(nn.Module):
    r"""SPVAE approximates any stochastic process.

    SPVAE could be understood as an approximation of the
    Karhunen-Loeve Expansion of a centered stochastic process:

    $$f(s)=\sum_{j=1}^\infty\beta_j\phi_j(s)$$

    Where the encoder is $f_\text{enc}:(\mathbf{s},\mathbf{f})\to\beta$
    and the decoder is $f_\text{dec}:(\mathbf{s},\beta)\to\beta^\intercal\phi(\mathbf{s})$.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate samples it was trained on.

    Args:
        encoder: A module used to encode GP realizations and
            their hyperparamters.
        decoder: A module used to decode random vectors and
            GP hyperparameters into GP samples.
        z_dim: The size of the hidden dimension.

    Returns:
        An instance of the `SPVAE` network.
    """

    encoder: nn.Module
    decoder: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, rng: Array, s: Array, f: Array, training: bool = False):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s: A location array of shape `(B,K,D)` where
                `B` is batch size, `K` is number of locations,
                and `D` is the dimension of each location.
            f: A function value array of shape `(B, K)`.

        Returns:
            $\hat{\mathbf{f}}$, a recreation of the original$\mathbf{f}$,
            along with $\mu$ and $\log(\sigma^2)$, which are often used
            to calculate losses involving KL divergence.
        """
        batch_size = f.shape[0]
        s_flat = s.reshape(batch_size, -1)
        f_flat = f.reshape(batch_size, -1)
        latents = self.encoder(jnp.hstack([s_flat, f_flat]), training)
        mu = nn.Dense(self.z_dim)(latents)
        log_var = nn.Dense(self.z_dim)(latents)
        std = jnp.exp(log_var / 2)
        eps = random.normal(rng, log_var.shape)
        z = mu + std * eps
        f_hat = self.decoder(jnp.hstack([s_flat, z]), training)
        return f_hat.reshape(f.shape), mu, log_var
