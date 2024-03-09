#!/usr/bin/env python3
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, jit, random
from jax.typing import ArrayLike


@dataclass
class PiVAE(nn.Module):
    r"""PiVAE approximates a stochastic process.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate samples from the approximated process.

    Args:
        phi: An instance of the `Phi` class.
        encoder: A module used to encode GP realizations and
            their hyperparamters.
        decoder: A module used to decode random vectors and
            GP hyperparameters into GP samples.
        beta_dim: The size of the hidden dimension.

    Returns:
        An instance of a `PiVAE` network.
    """

    phi: nn.Module
    encoder: nn.Module
    decoder: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, rng: Array, s: Array, f: Array):
        batch_size = s.shape[0]
        f_flat = f.reshape(batch_size, -1)
        s_flat = s.reshape(batch_size, -1)
        # TODO(danj): finish
        # |phi(s)| = num features per location
        # K tuples of (s, f) per row
        # Thus phi(s) for each row should be K|phi(s)|
        # Multiply betas.T x |phi(s)| for each of K locations, s
        phi_s = self.phi(s_flat)
        # beta vector for each row of the batch
        betas = self.param(
            "betas",
            nn.initializers.lecun_normal(),
            (batch_size, s_flat.shape[-1]),
        )
        latents = self.encoder(betas.weights)
        mu = nn.Dense(self.z_dim)(latents)
        log_var = nn.Dense(self.z_dim)(latents)
        std = jnp.exp(log_var / 2)
        eps = random.normal(rng, log_var.shape)
        z = mu + std * eps
        beta_hats = self.decoder(z)
        return f_hat.reshape(f.shape), mu, log_var, beta_hats


@dataclass
class Phi(nn.Module):
    r"""`Phi` approximates a collection of basis functions.

    `Phi` approximates the collection of basis functions in the
    Karhunen-Loeve Expansion of a centered stochastic process:

    $$f(s)=\sum_{j=1}^\infty\beta_j\phi_j(s)$$

    Args:
        dims: The number of dimensions for each layer. The
            first layer of the network is an RBF network
            and the remaining are linear.
        act_fn: The activation function for hidden layers.
        var: Variance for RBF kernel.

    Returns:
        An instance of the `Phi` network.
    """

    dims: list[int]
    act_fn: Callable = nn.relu
    var: float = 1.0

    @nn.compact
    def __call__(self, x):
        centers = self.param(
            "centers",
            nn.initializers.lecun_normal(),
            (self.dims[0], x.shape[-1]),
        )
        x = jnp.exp(-0.5 * l2_dist_sq(centers, x) / self.var)
        for dim in self.dims[1:-1]:
            x = nn.Dense(dim)(x)
            x = self.act_fn(x)
        return nn.Dense(self.dims[-1])(x)


@jit
def l2_dist_sq(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """L2 distance between two [..., D] arrays.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Matrix of all pairwise distances.
    """
    x, y = _prepare_dims(x, y)
    return (x**2).sum(-1)[:, None] + (y**2).sum(-1).T - 2 * x @ y.T


@jit
def _prepare_dims(x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Prepares dims for use in kernel functions.

    Args:
        x: Input array of size `[..., D]`.
        y: Input array of size `[..., D]`.

    Returns:
        Two `[N, D]` dimensional arrays.
    """
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    if y.ndim == 1:
        y = y[:, jnp.newaxis]
    x = x.reshape(-1, x.shape[-1])
    y = y.reshape(-1, y.shape[-1])
    return x, y
