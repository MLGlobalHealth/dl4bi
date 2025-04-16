from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers as init

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
    embed_dim: int
    num_latents: int

    @nn.compact
    def __call__(self, x: jax.Array):
        Z, E, D = self.num_latents, self.embed_dim, x.shape[-1]
        latent_tokens = self.param("latent_tokens", init.truncated_normal(), (Z, E))
        latent_inputs = self.param("latent_inputs", init.truncated_normal(), (Z, D))

        pass


class TNPDecoder(nn.Module):
    @nn.compact
    def __call__(self):
        pass
