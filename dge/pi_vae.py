#!/usr/bin/env python3
from collections.abc import Callable

import jax.numpy as jnp
from flax import linen as nn
from jax import Array, jit, random
from jax.typing import ArrayLike


class PiVAE(nn.Module):
    r"""PiVAE approximates a stochastic process.

    Once trained, the module's `decoder` can be used as a generative
    model to simulate samples from the approximated process.

    Warning:
        This model assumes that the total number of samples is the
        same as batch size, so it sees the same samples over and over.

    Args:
        phi: An instance of the `Phi` class.
        encoder: Learns to encode random betas.
        decoder: Decodes random latent vectors to betas.

    Returns:
        An instance of a `PiVAE` network.
    """

    phi: nn.Module
    encoder: nn.Module
    decoder: nn.Module
    z_dim: int

    @nn.compact
    def __call__(self, rng: Array, s: Array, f: Array):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s: A location array of shape `(B,L,D)` where
                `B` is batch size, `L` is number of locations,
                and `D` is the dimension of each location.
            f: A function value array of shape `(B, K)`.

        Returns:
            $\hat{\mathbf{f}}_\beta=\beta^\intercal\phi(\mathbf{s})$, $\hat{f}
            _{\hat{\beta}}=\hat{\beta}^\intercal\phi(\mathbf{s})$, $\mu_z$, and
            $\log(\sigma_z^2)$.
        """
        B, L, D = s.shape  # B=batch, L=num locations, D=location dim
        phi_s = self.phi(s.reshape(-1, D)).reshape(B, L, -1)  # BxLxF
        F = phi_s.shape[-1]  # F=|phi(s_i)|, the feature dimensionality
        betas = self.param("betas", nn.initializers.lecun_normal(), (B, F))
        f_hat_beta = jnp.einsum("BF,BLF->BL", betas, phi_s)
        # NOTE: When training the VAE, the betas should be fixed, i.e. you
        # do not want the input betas being updated via backprop because they
        # weren't close enough to the predicted beta_hats. To do this, you need
        # to use the frozen beta parameters, not the traced jax arrays.
        betas_fixed = self.variables["params"]["betas"]
        latents = self.encoder(betas_fixed)
        latents = self.encoder(betas)
        mu = nn.Dense(self.z_dim)(latents)
        log_var = nn.Dense(self.z_dim)(latents)
        std = jnp.exp(log_var / 2)
        eps = random.normal(rng, log_var.shape)
        z = mu + std * eps
        beta_hats = self.decoder(z)
        f_hat_beta_hat = jnp.einsum("BF,BLF->BL", beta_hats, phi_s)
        return f_hat_beta, f_hat_beta_hat, mu, log_var


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
        # RBF layer
        x = jnp.exp(-0.5 * l2_dist_sq(x, centers) / self.var)
        # Linear layers
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
