from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core import MLP, KRStack, LearnableEmbedding


class TNPDS(nn.Module):
    """A Transformer Neural Process - Diagonal (TNP-D) Simplified.

    Args:
        embed_s_f: A module or combining embedded locations and function values.
        enc: An encoder module for observed points.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `TNP-D` model.
    """

    embed_s_f: nn.Module = LearnableEmbedding(lambda x: x, MLP([64] * 4))
    enc: nn.Module = KRStack(num_blks=6, d_ffn=128)
    head: nn.Module = MLP([128] * 2 + [2])

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: A location array of shape `[B, S_ctx, D_S]` where
                `B` is batch size, `S_ctx` is number of context
                locations, and `D_S` is the dimension of each location.
            f_ctx: A function value array of shape `[B, S_ctx, D_F]` where `B` is
                batch size, `S_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `[B, S_test, D_S]` where `B` is
                batch size, `S_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens_ctx: An optional array of shape `(B,)` indicating the
                valid positions for each `S_ctx` sequence in the batch.
            valid_lens_test: An optional array of shape `(B,)` indicating the
                valid positions for each `S_test` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times S_\text{test}\times 2D_F}$.
        """
        B, L_ctx, _ = s_ctx.shape
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L_ctx, B)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], axis=-1)
        s_f_test = jnp.concatenate([s_test, f_test], axis=-1)
        s_f_ctx_embed = self.embed_s_f(s_f_ctx, training)
        s_f_test_embed = self.embed_s_f(s_f_test, training)
        s_f_test_enc, _ = self.enc(
            s_f_test_embed,
            s_f_ctx_embed,
            valid_lens_ctx,
            training,
            **kwargs,
        )
        f_dist = self.head(s_f_test_enc, training)
        f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
        return f_mu, jnp.exp(f_std)
