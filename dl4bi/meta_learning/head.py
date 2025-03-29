import flax.linen as nn
import jax
import jax.numpy as jnp

from dl4bi.core.mlp import MLP
from dl4bi.core.transformer import TransformerEncoder


class MVNHead(nn.Module):
    """
    Multivariate normal projection head using a low rank decomposition L = lower(HH^T).

    Taken from the implementation of TNP-ND.
    """

    # TODO @pgrynfelder remove this after all?

    dec_mu: nn.Module = MLP([128, 1])
    dec_std = TransformerEncoder()
    proj_std = MLP([128, 128, 128, 20])
    min_std = 0.0

    def __call__(
        self,
        f_enc: jax.Array,  # [B, L, D_F],
        valid_lens: jax.Array | None,  # [B]
        training: bool = False,
    ):
        B, L, D_f = f_enc.shape
        mu = self.dec_mu(f_enc, training)

        H = self.dec_std(f_enc, valid_lens, training)
        H = self.proj_std(H, valid_lens, training)
        H = H.reshape(B, L * D_f, -1)  # will also model dependencies across dims

        L = jnp.tril(H @ H.mT)

        return mu, L
