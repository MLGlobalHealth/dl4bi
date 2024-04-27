from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from .attention import MultiheadAttention
from .embed import FixedSinusoidalEmbedding
from .mlp import MLP
from .transformer import TransformerEncoder


# TODO(danj): incorporate valid_lens
# TODO(danj): try different global pooling mechanisms (max-pooling instead of mean)
class AttentiveNeuralProcess(nn.Module):
    embed_s: nn.Module = FixedSinusoidalEmbedding(128)
    enc_s_and_f_local: nn.Module = TransformerEncoder(FixedSinusoidalEmbedding(64))
    enc_s_and_f_global: nn.Module = TransformerEncoder(FixedSinusoidalEmbedding(64))
    cross_attn: nn.Module = MultiheadAttention()
    dec_z_mu: nn.Module = MLP([128, 128])
    dec_z_log_var: nn.Module = MLP([128, 128])
    dec_f_mu: nn.Module = MLP([128 * 3, 128 * 2, 128, 1])
    dec_f_log_var: nn.Module = MLP([128 * 3, 128 * 2, 128, 1])

    @nn.compact
    def __call__(
        self,
        key: jax.Array,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, S_ctx]
        training: bool = False,
    ):
        qs, ks = self.embed_s(s_test), self.embed_s(s_ctx)
        s_and_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        # local ("deterministic") path
        vs_local = self.enc_s_and_f_local(s_and_f_ctx, valid_lens, training)
        ctx_local, _ = self.cross_attn(qs, ks, vs_local, valid_lens, training)
        # global ("latent") path
        vs_global = self.enc_s_and_f_global(s_and_f_ctx, valid_lens, training)
        vs_global_mean = vs_global.mean(axis=1)
        z_mu = self.dec_z_mu(vs_global_mean, training)
        z_log_var = self.dec_z_log_var(vs_global_mean, training)
        zs_global = z_mu + jnp.exp(z_log_var / 2) * random.normal(key, z_mu.shape)
        # decoding context to (mu_f, log_var_f) for every test location
        zs_global = jnp.repeat(zs_global[:, jnp.newaxis, :], qs.shape[1], axis=1)
        ctx = jnp.concatenate([qs, ctx_local, zs_global], -1)
        f_mu = self.dec_f_mu(ctx)
        f_log_var = self.dec_f_log_var(ctx)
        return zs_global, f_mu, f_log_var
