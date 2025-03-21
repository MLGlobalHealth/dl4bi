import flax.linen as nn
from jraph import GraphsTuple

from .attention import MultiHeadGraphAttention
from .bias import RBFNetworkBias
from .mlp import MLP


class GraphAttentionBlock(nn.Module):
    attn: nn.Module = MultiHeadGraphAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([256, 64], nn.gelu)
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, g: GraphsTuple, training: bool, **kwargs):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        n_0, *_ = g
        n_1, _ = self.attn(g._replace(nodes=self.norm(n_0)), training, **kwargs)
        n_2 = n_0 + drop(n_1)
        n_3 = self.ffn(self.norm.copy()(n_2), training)
        return g._replace(nodes=n_2 + drop(n_3))
