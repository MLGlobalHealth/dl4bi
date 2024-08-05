from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from dsp.core.attention import MultiheadAttention

from ..core import (
    MLP,
    AddNorm,
)


class DKR(nn.Module):
    """Deep Kernel Regression.

    Args:
        num_layers: Number of attention layers.
        num_repeats: Number of times to repeat each attention layer.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_s_f: A module that jointly embeds the (embedded) index set and
            function values.
        attn: An attention module.
        head: A prediction head for decoded output.
        min_std: Minimum pointwise standard deviation.

    Returns:
        An instance of the `DKR` model.
    """

    num_layers: int = 3
    num_repeats: int = 2
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_s_f: nn.Module = MLP([64] * 4)
    attn: nn.Module = MultiheadAttention()
    head: nn.Module = MLP([128, 2])
    min_std: float = 0.0

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_S]
        f_ctx: jax.Array,  # [B, L_ctx, D_F]
        s_test: jax.Array,  # [B, L_test, D_S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: A location array of shape `[B, L_ctx, D_S]` where
                `B` is batch size, `L_ctx` is number of context
                locations, and `D_S` is the dimension of each location.
            f_ctx: A function value array of shape `[B, L_ctx, D_F]` where `B` is
                batch size, `L_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `[B, L_test, D_S]` where `B` is
                batch size, `L_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens_ctx: An optional array of shape `(B,)` indicating the
                valid positions for each `L_ctx` sequence in the batch.
            valid_lens_test: An optional array of shape `(B,)` indicating the
                valid positions for each `L_test` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\sigma_f\in\mathbb{R}^{B\times L_\text{test}\times D_F}$.
        """
        add_norm = AddNorm(0.0)
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        s_f_ctx = stack(self.embed_s(s_ctx), self.embed_f(f_ctx))
        s_f_test = stack(self.embed_s(s_test), self.embed_f(f_test))
        qvs = self.embed_s_f(s_f_test)
        kvs = self.embed_s_f(s_f_ctx)
        for i in range(self.num_layers):
            attn = self.attn.copy()
            _qvs, _kvs = qvs, kvs
            for _ in range(self.num_repeats):
                qvs, _ = attn(qvs, kvs, kvs, valid_lens_ctx, training)
                kvs, _ = attn(kvs, kvs, kvs, valid_lens_ctx, training)
                qvs = add_norm(_qvs, qvs)
                kvs = add_norm(_kvs, kvs)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        return f_mu, f_std
