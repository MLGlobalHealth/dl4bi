"""Conditional Neural Process model."""

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from .steps import likelihood_train_step, likelihood_valid_step


class CNP(nn.Module):
    """The Conditional Process as detailed in [Conditional Neural Processes](https://arxiv.org/pdf/1807.01613).


    This implementation is based on Google's official implementation [here]
    (https://github.com/google-deepmind/neural-processes/tree/master) and the
    hyperparameters follow Figure 8 on page 12 in [Attentive Neural Processes]
    (https://arxiv.org/abs/1901.05761) for comparison to the original Neural
    Process.

    Args:
        enc: A module for encoding context points.
        dec: A module for decoding at test points.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of `CNP`.
    """

    enc_det: nn.Module = MLP([128] * 6)
    dec: nn.Module = MLP([128] * 4 + [2])
    output_fn: Callable = DiagonalMVNOutput.from_activations
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        s_test: jax.Array,  # [B, L_test, D_s]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
        **kwargs,
    ):
        """Predict test outputs for a Conditional Neural Process."""
        r = self.encode_deterministic(s_ctx, f_ctx, mask_ctx, training)
        return self.decode(r, s_test, training)  # [B, n_z, L_test, d_f]

    def encode_deterministic(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_s]
        f_ctx: jax.Array,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
    ):
        """Pool context observations into a deterministic representation."""
        (B, L, _) = s_ctx.shape
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], -1)
        s_f_ctx_embed = self.enc_det(s_f_ctx, training)
        if mask_ctx is not None:
            return jnp.mean(s_f_ctx_embed, axis=1, where=mask_ctx[..., None])
        return jnp.mean(s_f_ctx_embed, axis=1)  # [B, d_ffn]

    def decode(
        self,
        r_ctx: jax.Array,  # [B, d_ffn]
        s_test: jax.Array,  # [B, L_test, D_s]
        training: bool = False,
    ):
        """Decode deterministic context features at test locations."""
        L_test = s_test.shape[1]
        r_ctx = jnp.repeat(r_ctx[:, None, :], L_test, axis=1)  # [B, L_test, d_ffn]
        q = jnp.concatenate([r_ctx, s_test], -1)  # [B, L_test, d_ffn + D_s]
        output = self.dec(q, training)
        return self.output_fn(output)
