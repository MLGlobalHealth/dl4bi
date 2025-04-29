from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from dl4bi.core.attention import MLP, MultiHeadAttention
from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.meta_learning.steps import elbo_train_step, likelihood_valid_step


class ANP(nn.Module):
    """The Attentive Neural Process as detailed in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761).

    This implementation is based on Google's official implementation
    [here](https://github.com/google-deepmind/neural-processes/tree/master)
    and the hyperparameters follow Figure 8 on page 12 in the paper.

    .. note::
        The paper does not indicate that there are any projection matrices for
        queries, keys, values in MultiHeadAttention, but does specify a linear
        projection for outputs. On the other hand, the code implementation
        uses a 2-layer MLP for queries and keys, and nothing for values or
        outputs. Here, we follow the standard MultiHeadAttention setup where all
        projection matrices are single layer linear projections.

    .. note::
        The paper specifies different MLPs and attention modules for the
        latent and deterministic paths, but the code implementation reuses the
        deterministic representation inside the latent path. Here, we follow
        the paper and compute both paths separately.

    Args:
        embed_s: An embedding module for locations.
        enc_det: An encoder for the deterministic path.
        enc_lat: An encoder for the latent path.
        self_attn_det: A self attention module for the deterministic path.
        self_attn_lat: A self attention module for the latent path.
        z_dist: A module that converts hidden representation to `z` mu and sigma.
        dec: A decoder for test locations.
        cross_attn: A cross attention module used in decoding.
        n_z: Number of latent `z` samples to use.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of an `ANP`.
    """

    embed_s: nn.Module = MLP([128] * 2)
    enc_det: nn.Module = MLP([128] * 3)
    enc_lat: nn.Module = MLP([128] * 3)
    self_attn_det: nn.Module = MultiHeadAttention(
        proj_qs=MLP([128]),
        proj_ks=MLP([128]),
        proj_vs=MLP([128]),
        proj_out=MLP([128]),
        num_heads=8,
    )
    self_attn_lat: nn.Module = MultiHeadAttention(
        proj_qs=MLP([128]),
        proj_ks=MLP([128]),
        proj_vs=MLP([128]),
        proj_out=MLP([128]),
        num_heads=8,
    )
    z_dist: nn.Module = MLP([128, 256])
    cross_attn: nn.Module = MultiHeadAttention(
        proj_qs=MLP([128]),
        proj_ks=MLP([128]),
        proj_vs=MLP([128]),
        proj_out=MLP([128]),
        num_heads=8,
    )
    dec: nn.Module = MLP([128] * 4 + [2])
    n_z: int = 1
    output_fn: Callable = DiagonalMVNOutput.from_latent_activations
    train_step: Callable = elbo_train_step
    valid_step: Callable = likelihood_valid_step
    norm: nn.Module = nn.LayerNorm()

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
        output = self.decode(r, z, s_ctx, s_test, mask_ctx, training)
        return output, DiagonalMVNOutput(z_mu_ctx, z_std_ctx)

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
    ):
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        s_f_ctx_embed = self.norm.copy()(s_f_ctx_embed)
        r_ctx, _ = self.self_attn_det(
            s_f_ctx_embed,
            s_f_ctx_embed,
            s_f_ctx_embed,
            mask_ctx,
            training,
        )
        return r_ctx

    def encode_latent(
        self,
        s_ctx: jax.Array,  # [B, L, D_s]
        f_ctx: jax.Array,  # [B, L, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, K]
        training: bool = False,
    ):
        (B, L, _) = s_ctx.shape
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_lat(s_f_ctx, training)
        s_f_ctx_embed = self.norm.copy()(s_f_ctx_embed)
        s_f_ctx_enc, _ = self.self_attn_lat(
            s_f_ctx_embed,
            s_f_ctx_embed,
            s_f_ctx_embed,
            mask_ctx,
            training,
        )
        if mask_ctx is None:
            s_f_ctx_means = jnp.mean(s_f_ctx_enc, axis=1)
        else:
            s_f_ctx_means = jnp.mean(s_f_ctx_enc, axis=1, where=mask_ctx[..., None])
        z_dist = self.z_dist(s_f_ctx_means, training)
        z_mu, z_std = jnp.split(z_dist, 2, axis=-1)
        z_std = 0.1 + 0.9 * nn.sigmoid(z_std)
        return z_mu, z_std  # [B, d_z]

    def decode(
        self,
        r_ctx: jax.Array,  # [B, d_ffn]
        z: jax.Array,  # [B, n_z, d_z]
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array],  # [B, K]
        training: bool = False,
    ):
        L_test = s_test.shape[1]
        r, _ = self.cross_attn(
            self.embed_s(s_test),  # qs
            self.embed_s(s_ctx),  # ks
            r_ctx,  # vs
            mask_ctx,
            training,
        )  # [B, L_test, d_ffn]
        r = jnp.repeat(r[:, None, ...], self.n_z, axis=1)  # [B, n_z, L_test, d_ffn]
        s = jnp.repeat(s_test[:, None, ...], self.n_z, axis=1)  # [B, n_z, L_test, D_s]
        z = jnp.repeat(z[..., None, :], L_test, axis=-2)  # [B, n_z, L_test, d_z]
        q = jnp.concatenate([s, r, z], -1)  # [B, n_z, L_test, D_s + d_ffn + d_z]
        output = self.dec(q, training)
        return self.output_fn(output)
