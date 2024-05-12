from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import MultiheadAttention
from .embed import LearnableEmbedding
from .mlp import MLP
from .transformer import TransformerEncoder


class HFTx(nn.Module):
    """A High Frequency Transformer (HFTx).

    Args:
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        enc: An encoder module for observed points.
        cross_attn: A cross-attention module for matching context and test points.
        dec: A decoder module for test points.

    Returns:
        An instance of the `HFTx` model.

    .. warning::
        `valid_lens` applies only to input context sequences, `s_ctx`. Test
        locations, `s_test`, are expected to be dense, i.e. not ragged or
        padded.
    """

    embed_s: nn.Module = LearnableEmbedding(lambda x: x, MLP([128] * 3))
    embed_s_f: nn.Module = MLP([128] * 3)
    enc: nn.Module = TransformerEncoder()
    cross_attn: nn.Module = MultiheadAttention()
    dec: nn.Module = MLP([128] * 3 + [2])

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, S_ctx, D_S]
        f_ctx: jax.Array,  # [B, S_ctx, D_F]
        s_test: jax.Array,  # [B, S_test, D_S]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, S_ctx]
        training: bool = False,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: A location array of shape `(B,S_ctx,D_S)` where
                `B` is batch size, `S_ctx` is number of context
                locations, and `D_S` is the dimension of each location.
            f_ctx: A function value array of shape `(B,S_ctx,D_F)` where `B` is
                batch size, `S_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `(B,S_test,D_S)` where `B` is
                batch size, `S_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens: An optional array of shape `(B,)` indicating the
                valid positions for each `S_ctx` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times S_\text{test}\times D_F}$.
        """
        qs, ks = self.embed_s(s_test, training), self.embed_s(s_ctx, training)
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.embed_s_f(s_f_ctx, training)
        vs = self.enc(s_f_ctx_embed, valid_lens, training)
        rs, _ = self.cross_attn(qs, ks, vs, valid_lens, training)
        params = self.dec(rs, training)
        f_mu, f_log_var = params[..., [0]], params[..., [1]]
        return f_mu, f_log_var
