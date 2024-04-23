"""
Transformer architecture largely based on [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from typing import Optional

import flax.linen as nn
import jax

from dge.embed import FixedSinusoidalEmbedding

from .attention import Attention, DotScorer, MultiheadAttention


class AddNorm(nn.Module):
    p_dropout: float

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, training: bool = False):
        y = nn.Dropout(self.p_dropout, deterministic=not training)(y)
        return nn.LayerNorm()(x + y)


class TransformerEncoderBlock(nn.Module):
    num_heads: int = 4
    p_dropout: float = 0.3
    attention: nn.Module = MultiheadAttention()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d = x.shape[-1]
        ctx, attn = self.attention(x, x, x, valid_lens, training)
        y = AddNorm(self.p_dropout)(x, ctx, training)
        ctx = nn.Sequential([nn.Dense(d), nn.relu, nn.Dense(d)])(y)
        return AddNorm(self.p_dropout)(y, ctx, training)


class TransformerEncoder(nn.Module):
    embed: nn.Module = FixedSinusoidalEmbedding()
    num_heads: int = 4
    num_blks: int = 3
    p_dropout: float = 0.3

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        x = self.embed(x)
        for _ in range(self.num_blks):
            x = TransformerEncoderBlock(self.num_heads, self.p_dropout)(
                x, valid_lens, training
            )
        return x
