from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core.mlp import MLP
from ..core.transformer import TransformerEncoder
from .model_output import TrilMVNOutput


class TNPND(nn.Module):
    """A Transformer Neural Process - Non-Diagonal (TNP-ND).

    .. note::
        This implements the 'Cholesky' decomposition version of TNP-ND.

    Args:
        embed_s_f: A module that embeds positions and function values.
        enc: An encoder module for observed points.
        dec_f_mu: A module that decodes the mean for function values.
        def_f_std: A module for decoding the lower triangular covariance
            matrix.
        proj_f_std: A module for projecting embeddings into a smaller
            vector space for use in computing a lower triangular covariance
            matrix.
        min_std: Used to bound the diagonal of the lower triangular covariance.

    Returns:
        An instance of the `TNP-ND` model.
    """

    embed_s_f: nn.Module = MLP([64] * 4)
    enc: nn.Module = TransformerEncoder()
    dec_f_mu: nn.Module = MLP([128, 1])
    dec_f_std: nn.Module = TransformerEncoder()
    proj_f_std: nn.Module = MLP([128] * 3 + [20])
    min_std: float = 0.0

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
        (B, L_ctx), (L_test, d_f) = s_ctx.shape[:2], f_ctx.shape[-2:]
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], d_f])
        s_f_test = jnp.concat([s_test, f_test], axis=-1)
        s_f = jnp.concat([s_f_ctx, s_f_test], axis=1)
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
        s_f_embed = self.embed_s_f(s_f, training)
        s_f_enc = self.enc(s_f_embed, mask, training, **kwargs)
        s_f_test_enc = s_f_enc[:, -L_test:, ...]
        f_mu = self.dec_f_mu(s_f_test_enc, training)
        f_std = self.dec_f_std(s_f_test_enc, None, training)
        f_std = self.proj_f_std(f_std, training).reshape(B, L_test * d_f, -1)
        f_L = jnp.tril(jnp.einsum("B I D, B J D -> B I J", f_std, f_std))
        # WARNING: using min_std can cause instability when solving the system
        # of equations in order to calculate the log pdf of the MVN
        if self.min_std:
            d = jnp.arange(L_test * d_f)
            f_L = f_L.at[:, d, d].set(
                # NOTE: tanh works since diag(f_std @ f_std.T) > 0
                self.min_std + (1 - self.min_std) * nn.tanh(f_L[:, d, d])
            )
        return TrilMVNOutput(f_mu, f_L)
