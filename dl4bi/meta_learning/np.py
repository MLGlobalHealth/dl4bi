from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from ..core.mlp import MLP
from .model_output import DiagonalMVNOutput
from .steps import elbo_train_step, likelihood_valid_step


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
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of `NP`.
    """

    enc_det: nn.Module = MLP([128] * 6)
    enc_lat: nn.Module = MLP([128] * 3)
    z_dist: nn.Module = MLP([128, 256])
    dec: nn.Module = MLP([128] * 4 + [2])
    n_z: int = 1
    output_fn: Callable = DiagonalMVNOutput.from_latent_np
    train_step: Callable = elbo_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
        **kwargs,
    ):
        r = self.encode_deterministic(s_ctx, f_ctx, mask_ctx, training)
        z_mu_ctx, z_std_ctx = self.encode_latent(s_ctx, f_ctx, mask_ctx, training)
        rng_z, z_shape = self.make_rng("extra"), (self.n_z, *z_mu_ctx.shape)
        z = z_mu_ctx + z_std_ctx * random.normal(rng_z, z_shape)  # [n_z, B, d_z]
        z = z.swapaxes(0, 1)  # [B, n_z, d_z]
        output = self.decode(r, z, s_test, training)  # [B, L_test, d_f]
        return self.output_fn(output), DiagonalMVNOutput(z_mu_ctx, z_std_ctx)

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        mask_ctx: Optional[jax.Array],  # [B, K]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        if mask_ctx is not None:
            return jnp.mean(s_f_ctx_embed, axis=1, where=mask_ctx[..., None])
        return jnp.mean(s_f_ctx_embed, axis=1)

    def encode_latent(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_lat(s_f_ctx, training)
        if mask_ctx is None:
            s_f_ctx_means = jnp.mean(s_f_ctx_embed, axis=1)
        else:
            s_f_ctx_means = jnp.mean(s_f_ctx_embed, axis=1, where=mask_ctx[..., None])
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
        return self.dec(q, training)
