from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

from ..core import MLP, mask_from_valid_lens


class CNP(nn.Module):
    """The Conditional Process as detailed in [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613).


    This implementation is based on Google's official implementation [here]
    (https://github.com/google-deepmind/neural-processes/tree/master) and the
    hyperparameters follow Figure 8 on page 11 in [Attentive Neural Processes]
    (https://arxiv.org/abs/1901.05761) for comparison to the original Neural
    Process.

    Args:
        d_ffn: The hidden dimension for all MLPs.

    Returns:
        An instance of `CNP`.
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
        f_mu, f_std = self.decode(r, s_test, d_f, training)  # [B, n_z, L_test, d_f]
        return f_mu, f_std

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

    def decode(
        self,
        r: jax.Array,  # [B, d_ffn]
        s_test: jax.Array,  # [B, L_test, D_s]
        d_f: int,
        training: bool = False,
    ):
        L_test = s_test.shape[1]
        r = jnp.repeat(r[:, None, :], L_test, axis=1)  # [B, L_test, d_ffn]
        q = jnp.concatenate([r, s_test], -1)  # [B, L_test, d_ffn + D_s]
        f_dist = MLP([self.d_ffn] * 4 + [2 * d_f])(q, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        # used in original implementation to prevent collapse
        f_std = 0.1 + 0.9 * nn.softplus(f_std)
        return f_mu, f_std  # [B, n_z, L_test, d_f]
