"""Translation-equivariant Transformer Neural Process model."""

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.transformer import TEISTEncoder
from ..core.utils import safe_stack
from .steps import likelihood_train_step, likelihood_valid_step
from .utils import first_shape


class TETNP(nn.Module):
    """[A Translation Equivariant Transformer Neural Process](https://arxiv.org/abs/2406.12409).

    .. note::
        By default, this uses an Induced Set Transformer (IST), which the
        authors noted performs best when using pseudo tokens.
    """

    encoder: Callable = TEISTEncoder()
    decoder: Callable = MLP([128, 128, 2])
    embed_f: Callable = MLP([128, 128])
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
        """Predict test outputs with the translation-equivariant TNP."""
        test_shape = first_shape([x_test, s_test, t_test])
        (B, L_ctx, _D_f), L_test = f_ctx.shape, test_shape[1]
        obs_ind, unobs_ind = jnp.ones((B, L_ctx, 1)), jnp.zeros((B, L_test, 1))
        f_test = jnp.zeros((*test_shape[:-1], f_ctx.shape[-1]))
        f_ctx = jnp.concat([f_ctx, obs_ind], axis=-1)
        f_test = jnp.concat([f_test, unobs_ind], axis=-1)
        f_ctx_embed, f_test_embed = self.embed_f(f_ctx), self.embed_f(f_test)
        # NOTE: All inputs other than the function value are considered part of
        # the index set for translation equivariance (see Appendix F.4).
        s_ctx = safe_stack(x_ctx, s_ctx, t_ctx)
        s_test = safe_stack(x_test, s_test, t_test)
        f_test_enc = self.encoder(f_test_embed, f_ctx_embed, s_test, s_ctx, mask_ctx)
        output = self.decoder(f_test_enc)
        return self.output_fn(output)
