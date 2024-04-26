import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import Attention
from .embed import FixedSinusoidalEmbedding
from .transformer import TransformerEncoder


# TODO(danj): embed target values as well? or target + loc values together?
# TODO(danj): add valid_lens for ragged input
class AttentiveNeuralProcess(nn.Module):
    s_embedder: nn.Module = FixedSinusoidalEmbedding()
    s_and_f_embedder: nn.Module = TransformerEncoder()
    attention: nn.Module = Attention()

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, N_ctx, D_S]
        f_ctx: jax.Array,  # [B, N_ctx, D_F]
        s_test: jax.Array,  # [B, N_test, D_S]
        training: bool = False,
    ):
        qs = self.s_embedder(s_test)
        ks = self.s_embedder(s_ctx)
        if s_ctx.ndim > f_ctx.ndim:
            f_ctx = f_ctx[..., None]
        vs = self.s_and_f_embedder(jnp.concatenate([s_ctx, f_ctx], -1))
        rs = Attention()(qs, ks, vs)
