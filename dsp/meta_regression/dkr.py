from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from dsp.core.attention import MultiheadAttention

from ..core import (
    MLP,
    AddNorm,
    LearnableEmbedding,
)


class DKR(nn.Module):
    """Deep Kernel Regression.

    Args:
        num_layers: Number of attention layers.
        num_repeats: Number of times to repeat each attention layer.
        embed_s: An embedding module for locations.
        embed_s_f: A module or combining embedded locations and function values.
        attn: An attention module.
        head: A prediction head for decoded output.
        add_norm: An `AddNorm` module applied between layers.

    Returns:
        An instance of the `DKR` model.
    """

    num_layers: int = 3
    num_repeats: int = 2
    embed_s: nn.Module = LearnableEmbedding(post_process=MLP([64, 64]))
    embed_s_f: nn.Module = MLP([64])
    attn: nn.Module = MultiheadAttention()
    head: nn.Module = MLP([64] * 2 + [2])
    add_norm: nn.Module = AddNorm(0.0)

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
        ks = self.embed_s(s_ctx, training)
        qvs = self.embed_s(s_test, training)
        kvs = self.embed_s_f(jnp.concatenate([ks, f_ctx], -1), training)
        for i in range(self.num_layers):
            attn = self.attn.copy()
            _qvs, _kvs = qvs, kvs
            for _ in range(self.num_repeats):
                qvs, _ = attn(qvs, kvs, kvs, valid_lens_ctx, training)
                kvs, _ = attn(kvs, kvs, kvs, valid_lens_ctx, training)
            if i + 1 != self.num_layers:  # add_norm all but last layer
                qvs = self.add_norm(_qvs, qvs)
                kvs = self.add_norm(_kvs, kvs)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        return f_mu, f_std
