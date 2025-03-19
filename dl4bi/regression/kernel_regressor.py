import flax.linen as nn
import jax

from ..core.attention import Attention
from ..core.embed import FixedSinusoidalEmbedding


class KernelRegressor(nn.Module):
    location_embedder: nn.Module = FixedSinusoidalEmbedding()
    p_dropout: float = 0.0

    def setup(self):
        self.attn = Attention(self.p_dropout)
        embed_dim = self.location_embedder.embed_dim
        self.W_q = nn.Dense(embed_dim)
        self.W_k = nn.Dense(embed_dim)

    def __call__(
        self,
        s_ctx: jax.Array,
        f_ctx: jax.Array,
        s_test: jax.Array,
        training: bool = False,
    ):
        qs = self.W_q(self.location_embedder(s_test))
        ks = self.W_k(self.location_embedder(s_ctx))
        vs, _ = self.attn(qs, ks, f_ctx, training=training)
        return vs
