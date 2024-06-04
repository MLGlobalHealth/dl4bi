from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from ..core import MLP, mask_from_valid_lens


class NP(nn.Module):
    """The original Neural Process as detailed in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761).

    This implementation is based on Google's official implementation
    [here](https://github.com/google-deepmind/neural-processes/tree/master)
    and the hyperparameters follow Figure 8 on page 11 in the paper.

    Args:
        d_ffn: The hidden dimension for all MLPs.
        d_z: The latent hidden dimension.
        n_z: Number of latent `z` samples to use.

    Returns:
        An instance of `NP`.
    """

    d_ffn: int = 128
    d_z: int = 128
    n_z: int = 1

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        d_f = f_ctx.shape[-1]
        r = self.encode_deterministic(s_ctx, f_ctx, valid_lens_ctx, training)
        z_mu_ctx, z_std_ctx = self.encode_latent(s_ctx, f_ctx, valid_lens_ctx, training)
        rng_z, z_shape = self.make_rng("latent_z"), (self.n_z, *z_mu_ctx.shape)
        z = z_mu_ctx + z_std_ctx * random.normal(rng_z, z_shape)  # [n_z, B, d_z]
        z = z.swapaxes(0, 1)  # [B, n_z, d_z]
        f_mu, f_std = self.decode(r, z, s_test, d_f, training)  # [B, n_z, L_test, d_f]
        return f_mu, f_std, z_mu_ctx, z_std_ctx

    def encode_deterministic(
        self,
        s: jax.Array,  # [B, L, D_s]
        f: jax.Array,  # [B, L, D_f]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        (B, L, _) = s.shape
        if valid_lens is None:
            valid_lens = jnp.repeat(L, B)
        mask = mask_from_valid_lens(L, valid_lens)
        s_f = jnp.concatenate([s, f], -1)
        s_f_embed = MLP([self.d_ffn] * 6)(s_f, training)
        return jnp.mean(s_f_embed, axis=1, where=mask)  # [B, d_ffn]

    def encode_latent(
        self,
        s: jax.Array,  # [B, L, D_s]
        f: jax.Array,  # [B, L, D_f]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        (B, L, _) = s.shape
        if valid_lens is None:
            valid_lens = jnp.repeat(L, B)
        mask = mask_from_valid_lens(L, valid_lens)
        s_f = jnp.concatenate([s, f], -1)
        s_f_embed = MLP([self.d_ffn] * 3)(s_f, training)
        s_f_means = jnp.mean(s_f_embed, axis=1, where=mask)
        z_dist = MLP([self.d_ffn, 2 * self.d_z])(s_f_means, training)
        z_mu, z_std = jnp.split(z_dist, 2, axis=-1)
        # used in original implementation to prevent collapse
        z_std = 0.1 + 0.9 * nn.sigmoid(z_std)
        return z_mu, z_std  # [B, d_z]

    def decode(
        self,
        r: jax.Array,  # [B, d_ffn]
        z: jax.Array,  # [B, n_z, d_z]
        s_test: jax.Array,  # [B, L_test, D_s]
        d_f: int,
        training: bool = False,
    ):
        L_test = s_test.shape[1]
        r = jnp.repeat(r[:, None, :], self.n_z, axis=1)  # [B, n_z, d_ffn]
        r_z = jnp.concatenate([r, z], -1)  # [B, n_z, d_ffn + d_z]
        r_z = jnp.expand_dims(r_z, 2)  # [B, n_z, 1, d_ffn + d_z]
        r_z = jnp.repeat(r_z, L_test, axis=2)  # [B, n_z, L_test, d_ffn + d_z]
        s = jnp.repeat(s_test[:, None, ...], self.n_z, axis=1)  # [B, n_z, L_test, D_s]
        q = jnp.concatenate([r_z, s], -1)  # [B, n_z, L_test, d_ffn + d_z + D_s]
        f_dist = MLP([self.d_ffn] * 4 + [2 * d_f])(q, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        # used in original implementation to prevent collapse
        f_std = 0.1 + 0.9 * nn.softplus(f_std)
        return f_mu, f_std  # [B, n_z, L_test, d_f]
