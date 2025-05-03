from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.transformer import TransformerEncoder
from ..core.utils import safe_stack
from .steps import likelihood_train_step, likelihood_valid_step
from .utils import first_shape


class TNPD(nn.Module):
    """A Transformer Neural Process - Diagonal (TNP-D).

    Args:
        embed_s_f: A module that embeds positions and function values.
        enc: An encoder module for observed points.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `TNP-D` model.
    """

    embed_s_f: nn.Module = MLP([64] * 4)
    enc: nn.Module = TransformerEncoder()
    head: nn.Module = MLP([128, 2], nn.relu)
    output_fn: Callable = DiagonalMVNOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        x_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_x]
        s_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
        t_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_t]
        f_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        x_test: Optional[jax.Array] = None,  # [B, L_test, D_x]
        s_test: Optional[jax.Array] = None,  # [B, L_test, D_s]
        t_test: Optional[jax.Array] = None,  # [B, L_test, D_t]
        training: bool = False,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            x_ctx: Optional fixed effects for context points.
            t_ctx: Optional temporal values for context points.
            s_ctx: Optional spatial values for context points.
            f_ctx: Function values for context points.
            mask_ctx: A mask for context points.
            x_test: Optional fixed effects for test points.
            t_test: Optional temporal values for test points.
            s_test: Optional spatial values for test points.
            f_test: Function values for test points.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            `ModelOutput`.
        """
        (B, L_ctx), L_test = s_ctx.shape[:2], s_test.shape[1]
        test_shape = first_shape([x_test, s_test, t_test])
        f_test = jnp.zeros((*test_shape[:-1], f_ctx.shape[-1]))
        # NOTE: TNP-D does differentiate between fixed effects
        # locations, and time, so stack them all together.
        ctx = safe_stack(x_ctx, s_ctx, t_ctx, f_ctx)
        test = safe_stack(x_test, s_test, t_test, f_test)
        ctx_test = jnp.concatenate([ctx, test], axis=1)
        ctx_test_embed = self.embed_s_f(ctx_test, training)
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
        ctx_test_enc = self.enc(ctx_test_embed, mask, training, **kwargs)
        test_enc = ctx_test_enc[:, -L_test:, ...]
        output = self.head(test_enc, training)
        return self.output_fn(output)
