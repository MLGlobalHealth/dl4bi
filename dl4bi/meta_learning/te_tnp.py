from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.transformer import TEISTEncoder
from .steps import likelihood_train_step, likelihood_valid_step


class TETNP(nn.Module):
    encoder: Callable = TEISTEncoder()
    decoder: Callable = MLP([128, 128, 2])
    embed_f: Callable = MLP([128, 128])
    output_fn: Callable = DiagonalMVNOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

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
        (B, L_ctx, D_f), L_test = f_ctx.shape, s_test.shape[1]
        obs_ind, unobs_ind = jnp.ones((B, L_ctx, 1)), jnp.zeros((B, L_test, 1))
        f_ctx = jnp.concat([f_ctx, obs_ind], axis=-1)
        f_test = jnp.concat([jnp.zeros((B, L_test, D_f)), unobs_ind], axis=-1)
        f_ctx_embed, f_test_embed = self.embed_f(f_ctx), self.embed_f(f_test)
        f_test_enc = self.encoder(f_test_embed, f_ctx_embed, s_test, s_ctx, mask_ctx)
        output = self.decoder(f_test_enc)
        return self.output_fn(output)
