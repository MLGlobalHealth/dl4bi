from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.transformer import TransformerEncoder
from .model_output import DiagonalMVNOutput


class TNPD(nn.Module):
    """A Transformer Neural Process - Diagonal (TNP-D).

    Args:
        embed_s_f: A module that embeds positions and function values.
        enc: An encoder module for observed points.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.

    Returns:
        An instance of the `TNP-D` model.
    """

    embed_s_f: nn.Module = MLP([64] * 4)
    enc: nn.Module = TransformerEncoder()
    head: nn.Module = MLP([128, 2], nn.relu)
    output_fn: Callable = DiagonalMVNOutput.from_conditional_np

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_S]
        f_ctx: jax.Array,  # [B, L_ctx, D_F]
        s_test: jax.Array,  # [B, L_test, D_S]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
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
            mask_ctx: An optional array of shape `[B, L_ctx]`
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times L_\text{test}\times D_F}$.
        """
        (B, L_ctx), L_test = s_ctx[:2], s_test.shape[1]
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        s_f_test = jnp.concatenate([s_test, f_test], axis=-1)
        s_f = jnp.concatenate([s_f_ctx, s_f_test], axis=1)
        s_f_embed = self.embed_s_f(s_f, training)
        if mask_ctx is None:
            mask_ctx = jnp.ones((B, L_ctx), dtype=bool)
            mask_test = jnp.zeros((B, L_test), dtype=bool)
            mask = jnp.concat([mask_ctx, mask_test], axis=1)
        else:
            mask = jnp.pad(
                mask_ctx,
                pad_width=((0, 0), (0, L_test)),
                mode="constant",
                constant_values=False,
            )
        s_f_enc = self.enc(s_f_embed, mask, training, **kwargs)
        s_f_test_enc = s_f_enc[:, -L_test:, ...]
        output = self.head(s_f_test_enc, training)
        return self.output_fn(output)
