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


class EdgeBiasedGAT(nn.Module):
    """An Edge Biased Graph Attention Network (EBGAT).

    Args:
        num_blks: Number of blocks to use.
        num_reps: Number of times to repeat each block.
        bias: A bias term to apply per block per repeat.
        blk: A graph convolution block.
    """

    num_blks: int = 6
    num_reps: int = 1
    bias: nn.Module = RBFNetworkBias()
    blk: nn.Module = GraphAttentionBlock()

    @nn.compact
    def __call__(
        self,
        g: GraphsTuple,
        training: bool = False,
        **kwargs,
    ):
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                bias = self.bias.copy()(g.edges, g.globals.get("edge_mask"))
                # NOTE: bucket_size is for numerical stability in
                # jax.ops.segment_* calls; this is typically only needed for
                # testing implementation correctness
                g = blk(g, training, bias=bias, bucket_size=kwargs.get("bucket_size"))
        return g
