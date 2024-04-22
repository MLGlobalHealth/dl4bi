#!/usr/bin/env python3
import jax.numpy as jnp
import numpy as np
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
        p_holdout: Probability of holding out a location from
            the encoding process. These locations are then
            inserted back in the decoding process to ensure
            the latent randomness generalizes to new locations.

    Returns:
        An instance of the `SPVAE` network.
    """

    encoder: nn.Module
    decoder: nn.Module
    z_dim: int
    p_holdout: float = 0.2

    @nn.compact
    def __call__(self, rng: Array, s: Array, f: Array, training: bool = False):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s: A location array of shape `(B,L,D)` where
                `B` is batch size, `L` is number of locations,
                and `D` is the dimension of each location.
            f: A function value array of shape `(B, L)`.

        Returns:
            $\mathbf{s}$ and $\mathbf{f}$ reordered along with $
            \hat{\mathbf{f}}$ a recreation of the reordered $\mathbf{f}$ and
            $\mu$ and $\log(\sigma^2)$, which are often used to calculate
            losses involving KL divergence.
        """
        B, L, _ = s.shape
        rng_keep, rng_z = random.split(rng)
        num_keep = L - int(L * self.p_holdout)
        keep_idx = random.choice(rng_keep, jnp.arange(L), (num_keep,), replace=False)
        # NOTE: this permutes the locations, since keep_idx isn't ordered
        s_keep_flat = s[:, keep_idx, :].reshape(B, -1)
        f_keep_flat = f[:, keep_idx, :].reshape(B, -1)
        latents = self.encoder(jnp.hstack([f_keep_flat, s_keep_flat]), training)
        mu = nn.Dense(self.z_dim)(latents)
        log_var = nn.Dense(self.z_dim)(latents)
        std = jnp.exp(log_var / 2)
        eps = random.normal(rng_z, log_var.shape)
        z = mu + std * eps
        # use all locations in the decoder
        s_holdout_flat = jnp.delete(
            s, keep_idx, axis=1, assume_unique_indices=True
        ).reshape(B, -1)
        f_holdout_flat = jnp.delete(
            f, keep_idx, axis=1, assume_unique_indices=True
        ).reshape(B, -1)
        s_flat = jnp.hstack([s_keep_flat, s_holdout_flat])
        f_flat = jnp.hstack([f_keep_flat, f_holdout_flat])
        f_hat_flat = self.decoder(
            jnp.hstack([z, s_keep_flat, s_holdout_flat]), training
        )
        return (
            s_flat.reshape(s.shape),
            f_flat.reshape(f.shape),
            f_hat_flat.reshape(f.shape),
            mu,
            log_var,
        )
