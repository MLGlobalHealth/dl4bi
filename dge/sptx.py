from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import MultiheadAttention
from .embed import LearnableEmbedding
from .mlp import MLP
from .transformer import TransformerDecoder, TransformerEncoder


class SPTx(nn.Module):
    """A Stochastic Process Transformer (SPTx).

    Args:
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        enc: An encoder module for observed points.
        dec: A decoder module for target points.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `SPTx` model.

    .. warning::
        `valid_lens` applies only to input context sequences, `s_ctx`. Test
        locations, `s_test`, are expected to be dense, i.e. not ragged or
        padded.
    """

    embed_s: nn.Module = LearnableEmbedding(lambda x: x, MLP([128] * 3))
    embed_s_f: nn.Module = MLP([128])
    enc: nn.Module = TransformerEncoder()
    dec: nn.Module = TransformerDecoder()
    head: nn.Module = MLP([128] * 2 + [2])

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, S_ctx]
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
            valid_lens: An optional array of shape `(B,)` indicating the
                valid positions for each `S_ctx` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times S_\text{test}\times D_F}$.
        """
        s_ctx_embed = self.embed_s(s_ctx, training)
        s_test_embed = self.embed_s(s_test, training)
        s_f_ctx = jnp.concatenate([s_ctx_embed, f_ctx], -1)
        s_f_ctx_embed = self.embed_s_f(s_f_ctx, training)
        s_f_enc = self.enc(s_f_ctx_embed, valid_lens, training, **kwargs)
        s_f_dec = self.dec(s_test_embed, s_f_enc, None, valid_lens, training, **kwargs)
        f_mu_log_var = self.head(s_f_dec, training)
        f_mu, f_log_var = f_mu_log_var[..., [0]], f_mu_log_var[..., [1]]
        return f_mu, f_log_var
