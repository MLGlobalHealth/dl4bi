from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .embed import LearnableEmbedding
from .mlp import MLP
from .transformer import TransformerEncoder


class TNPD(nn.Module):
    """A Transformer Neural Process - Diagonal (TNPD).

    Args:
        embed_s_f: A module or combining embedded locations and function values.
        enc: An encoder module for observed points.
        head: A prediction head for decoded output.

    Returns:
        An instance of the `TNPD` model.
    """

    embed_s_f: nn.Module = LearnableEmbedding(lambda x: x, MLP([128] * 4))
    enc: nn.Module = TransformerEncoder()
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
        (B, L_ctx, _), L_test = s_ctx.shape, s_test.shape[1]
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        f_test_z = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        s_f_test = jnp.concatenate([s_test, f_test_z], -1)
        s_f = _ragged_concat(s_f_ctx, s_f_test, valid_lens_ctx)
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(L_ctx, B)
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        valid_lens = valid_lens_ctx + valid_lens_test
        s_f_embed = self.embed_s_f(s_f, training)
        s_f_enc = self.enc(s_f_embed, valid_lens, training, **kwargs)
        f_dist = self.head(s_f_enc, training)
        f_dist = _ragged_extract(f_dist, valid_lens_ctx, L_test)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        return f_mu, f_log_var


def _ragged_concat(
    a: jax.Array,
    b: jax.Array,
    valid_lens_a: Optional[jax.Array] = None,
):
    if valid_lens_a is None:
        return jnp.concatenate([a, b], 1)
    L = b.shape[1]
    c = jnp.pad(a, ((0, 0), (0, L), (0, 0)))
    for i, j in enumerate(valid_lens_a):
        c = c.at[i, j : j + L, ...].set(b[i, ...])
    return c


def _ragged_extract(
    a: jax.Array,
    start_idx: jax.Array,
    width: int,
):
    B, _, D = a.shape
    b = jnp.zeros((B, width, D))
    for i, s in enumerate(start_idx):
        b = b.at[i, ...].set(a[i, s : s + width, ...])
    return b
