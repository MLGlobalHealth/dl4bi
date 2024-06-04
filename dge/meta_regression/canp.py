from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from ..core import MLP, MultiheadAttention


class CANP(nn.Module):
    """The Conditional Attentive Neural Process as detailed in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) and [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613).

    This implementation is based on Google's official implementation [here]
    (https://github.com/google-deepmind/neural-processes/tree/master) and the
    hyperparameters follow Figure 8 on page 11 in [Attentive Neural Processes]
    (https://arxiv.org/abs/1901.05761) for comparison to the original Neural
    Process.

    Args:
        d_ffn: The hidden dimension for all MLPs.

    Returns:
        An instance of an `CANP`.
    """

    d_ffn: int = 128

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
        f_mu, f_std = self.decode(
            r,
            s_ctx,
            s_test,
            valid_lens_ctx,
            d_f,
            training,
        )  # [B, n_z, L_test, d_f]
        return f_mu, f_std

    def encode_deterministic(
        self,
        s: jax.Array,  # [B, L, D_s]
        f: jax.Array,  # [B, L, D_f]
        valid_lens: Optional[jax.Array] = None,  # [B]
        training: bool = False,
    ):
        s_f = jnp.concatenate([s, f], -1)
        s_f_embed = MLP([self.d_ffn] * 3)(s_f, training)
        r, _ = MultiheadAttention()(
            s_f_embed,
            s_f_embed,
            s_f_embed,
            valid_lens,
            training,
        )
        return r

    def decode(
        self,
        r_ctx: jax.Array,  # [B, d_ffn]
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        s_test: jax.Array,  # [B, L_test, D_s]
        valid_lens_ctx: Optional[jax.Array],  # [B]
        d_f: int,
        training: bool = False,
    ):
        embed = MLP([self.d_ffn] * 2)
        r, _ = MultiheadAttention()(
            embed(s_test),  # qs
            embed(s_ctx),  # ks
            r_ctx,  # vs
            valid_lens_ctx,
            training,
        )  # [B, L_test, d_ffn]
        q = jnp.concatenate([r, s_test], -1)  # [B, L_test, d_ffn + D_s]
        f_dist = MLP([self.d_ffn] * 4 + [2 * d_f])(q, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        # used in original implementation to prevent collapse
        f_std = 0.1 + 0.9 * nn.softplus(f_std)
        return f_mu, f_std  # [B, n_z, L_test, d_f]
