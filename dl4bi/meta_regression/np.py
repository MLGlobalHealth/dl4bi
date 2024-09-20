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
    and the hyperparameters follow Figure 8 on page 12 in the paper.

    Args:
        enc_det: An encoder for the deterministic path.
        enc_lat: An encoder for the latent path.
        z_dist: A module that converts hidden representation to `z` mu and sigma.
        dec: A decoder for test locations.
        n_z: Number of latent `z` samples to use.
        min_std: Bounds standard deviation, default 0.0 (original 0.1).

    Returns:
        An instance of `NP`.
    """

    enc_det: nn.Module = MLP([128] * 6)
    enc_lat: nn.Module = MLP([128] * 3)
    z_dist: nn.Module = MLP([128, 256])
    dec: nn.Module = MLP([128] * 4 + [2])
    n_z: int = 1
    min_std: float = 0.0

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
        r = self.encode_deterministic(s_ctx, f_ctx, valid_lens_ctx, training)
        z_mu_ctx, z_std_ctx = self.encode_latent(s_ctx, f_ctx, valid_lens_ctx, training)
        rng_z, z_shape = self.make_rng("extra"), (self.n_z, *z_mu_ctx.shape)
        z = z_mu_ctx + z_std_ctx * random.normal(rng_z, z_shape)  # [n_z, B, d_z]
        z = z.swapaxes(0, 1)  # [B, n_z, d_z]
        f_mu, f_std = self.decode(r, z, s_test, training)  # [B, L_test, d_f]
        return f_mu, f_std, z_mu_ctx, z_std_ctx

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L, B)
        mask = mask_from_valid_lens(L, valid_lens_ctx)
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        return jnp.mean(s_f_ctx_embed, axis=1, where=mask)  # [B, d_ffn]

    def encode_latent(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L, B)
        mask = mask_from_valid_lens(L, valid_lens_ctx)
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_lat(s_f_ctx, training)
        s_f_ctx_means = jnp.mean(s_f_ctx_embed, axis=1, where=mask)
        z_dist = self.z_dist(s_f_ctx_means, training)
        z_mu, z_std = jnp.split(z_dist, 2, axis=-1)
        z_std = 0.1 + 0.9 * nn.sigmoid(z_std)
        return z_mu, z_std  # [B, d_z]

    def decode(
        self,
        r_ctx: jax.Array,  # [B, d_ffn]
        z: jax.Array,  # [B, n_z, d_z]
        s_test: jax.Array,  # [B, L_test, D_s]
        training: bool = False,
    ):
        L_test = s_test.shape[1]
        r_ctx = jnp.repeat(r_ctx[:, None, :], self.n_z, axis=1)  # [B, n_z, d_ffn]
        r_ctx_z = jnp.concatenate([r_ctx, z], -1)  # [B, n_z, d_ffn + d_z]
        r_ctx_z = jnp.expand_dims(r_ctx_z, 2)  # [B, n_z, 1, d_ffn + d_z]
        r_ctx_z = jnp.repeat(r_ctx_z, L_test, axis=2)  # [B, n_z, L_test, d_ffn + d_z]
        s = jnp.repeat(s_test[:, None, ...], self.n_z, axis=1)  # [B, n_z, L_test, D_s]
        q = jnp.concatenate([r_ctx_z, s], -1)  # [B, n_z, L_test, d_ffn + d_z + D_s]
        f_dist = self.dec(q, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        f_std = self.min_std + (1 - self.min_std) * nn.softplus(f_std)
        return f_mu.mean(axis=1), f_std.mean(axis=1)  # [B, L_test, d_f]
