from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from .steps import likelihood_train_step, likelihood_valid_step


class TETNP(nn.Module):
    encoder: Callable = TEISTEncoder()
    decoder: Callable = TNPDecoder()
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
        # preprocess_observations
        # yt is [[0 1]] for each entry
        # yc is [[v 0]] for each entry
        # embed f_ctx, f_test
        # encode
        pass


class TEISTEncoder(nn.Module):
    @nn.compact
    def __call__(self):
        pass


class TNPDecoder(nn.Module):
    @nn.compact
    def __call__(self):
        pass
