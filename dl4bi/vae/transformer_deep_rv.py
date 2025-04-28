import flax.linen as nn
import jax
import jax.numpy as jnp
from sps.kernels import l2_dist

from ..core.attention import MultiHeadAttention
from ..core.bias import Bias
from ..core.mlp import MLP
from ..core.transformer import TransformerEncoderBlock


class TransformerDeepRV(nn.Module):
    dim: int = 64
    num_blks: int = 2

    @nn.compact
    def __call__(self, z: jax.Array, ls: jax.Array, s: jax.Array, **kwargs):
        (B, L), D = z.shape, self.dim
        ids = jnp.repeat(jnp.arange(L, dtype=int)[None, :], B, axis=0)
        ids_embed = nn.Embed(L, features=D * 2 - 2)(ids)
        ls = jnp.tile(ls, (B, L, 1))
        x = jnp.concat([ids_embed, z[..., None], ls], axis=-1)
        x = MLP([D * 4, D], nn.gelu)(x)
        # TODO(danj): should cache this l2_dist(s, s)
        d = jnp.repeat(l2_dist(s, s)[None, ...], B, axis=0)
        for _ in range(self.num_blks):
            attn = MultiHeadAttention(
                proj_qs=MLP([D * 2]),
                proj_ks=MLP([D * 2]),
                proj_vs=MLP([D * 2]),
                proj_out=MLP([D]),
            )
            ffn = MLP([D * 4, D])
            bias = Bias.build_rbf_network_bias()(d)
            x, _ = TransformerEncoderBlock(attn=attn, ffn=ffn)(x, bias=bias)
        return MLP([D * 4, D, 1])(x)
