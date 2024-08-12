from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..core import MLP, TransformerEncoder


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
    enc: nn.Module = TransformerEncoder(num_blks=6, d_ffn=128)
    dec_f_mu: nn.Module = MLP([128, 1])
    dec_f_std: nn.Module = TransformerEncoder(num_blks=2, d_ffn=128)
    proj_f_std: nn.Module = MLP([128] * 3 + [20])
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
            $\mu_f,\log(\sigma_f^2\in\mathbb{R}^{B\times L_\text{test}\times D_F}$.
        """
        (B, L_test, _), d_f = s_test.shape, f_ctx.shape[-1]
        s_f_ctx = jnp.concatenate([s_ctx, f_ctx], axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], d_f])
        s_f_test = jnp.concatenate([s_test, f_test], axis=-1)
        s_f = jnp.concatenate([s_f_ctx, s_f_test], axis=1)
        s_f_embed = self.embed_s_f(s_f, training)
        s_f_enc = self.enc(s_f_embed, valid_lens_ctx, training, **kwargs)
        s_f_test_enc = s_f_enc[:, -L_test:, ...]
        f_mu = self.dec_f_mu(s_f_test_enc, training)
        f_std = self.dec_f_std(s_f_test_enc, valid_lens_test, training)
        f_std = self.proj_f_std(f_std, training).reshape(B, L_test * d_f, -1)
        f_L = jnp.tril(jnp.einsum("bid,bjd->bij", f_std, f_std))
        # WARNING: using min_std can cause instability when solving the system
        # of equations in order to calculate the log pdf of the MVN
        if self.min_std:
            d = jnp.arange(L_test * d_f)
            f_L = f_L.at[:, d, d].set(
                # NOTE: tanh works since diag(f_std @ f_std.T) > 0
                self.min_std + (1 - self.min_std) * nn.tanh(f_L[:, d, d])
            )
        return f_mu, f_L
