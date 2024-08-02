from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core import MLP, KRStack


class SPTx(nn.Module):
    """A Stochastic Process Transformer (SPTx).

    Args:
        embed_s: A module that embeds positions prior to embedding with
            function values.
        embed_s_f: A module that embeds positions and function values.
        dec: A decoder module, e.g. a `KRStack`.
        head: A prediction head for decoded output.
        min_std: Minimum standard deviation.

    Returns:
        An instance of the `SPTx` model.
    """

    embed_s: Callable = lambda x: x
    embed_s_f: nn.Module = MLP([64] * 4)
    dec: nn.Module = KRStack()
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
        f_test_zeros = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        s_ctx_embed, s_test_embed = self.embed_s(s_ctx), self.embed_s(s_test)
        s_f_ctx = jnp.concatenate([s_ctx_embed, s_ctx, f_ctx], axis=-1)
        s_f_test = jnp.concatenate([s_test_embed, s_test, f_test_zeros], axis=-1)
        s_f_test_enc, _ = self.dec(
            self.embed_s_f(s_f_test),
            self.embed_s_f(s_f_ctx),
            valid_lens_ctx,
            training,
        )
        f_dist = self.head(s_f_test_enc, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        return f_mu, f_std
