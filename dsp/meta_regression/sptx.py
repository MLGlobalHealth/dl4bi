from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core import MLP, KRStack, LearnableEmbedding


class SPTx(nn.Module):
    """A Stochastic Process Transformer (SPTx).

    Args:
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        dec: A decoder module, e.g. a `KRStack`.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `SPTx` model.
    """

    embed_s: nn.Module = LearnableEmbedding(post_process=MLP([64, 64, 64, 64]))
    embed_s_f: nn.Module = MLP([64])
    dec: nn.Module = KRStack()
    head: nn.Module = MLP([64, 64, 2])

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
            $\mu_f,\sigma_f\in\mathbb{R}^{B\times S_\text{test}\times 2D_F}$.
        """
        s_ctx_embed = self.embed_s(s_ctx, training)
        s_test_embed = self.embed_s(s_test, training)
        s_f_ctx = jnp.concatenate([s_ctx_embed, f_ctx], -1)
        s_f_ctx_embed = self.embed_s_f(s_f_ctx, training)
        s_f_test_enc, _ = self.dec(
            s_test_embed,
            s_f_ctx_embed,
            valid_lens_ctx,
            training,
        )
        f_dist = self.head(s_f_test_enc, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        return f_mu, f_std
