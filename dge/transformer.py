"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from typing import Optional

import flax.linen as nn
import jax

from dge.embed import FixedSinusoidalEmbedding

from .attention import Attention, DotScorer, MultiheadAttention


class AddNorm(nn.Module):
    """Performs add and norm from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        p_dropout: Dropout rate for input `y`.

    Returns:
        Add-and-normed input.
    """

    p_dropout: float

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, training: bool = False):
        y = nn.Dropout(self.p_dropout, deterministic=not training)(y)
        return nn.LayerNorm()(x + y)


class TransformerEncoderBlock(nn.Module):
    """A single encoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attention: Attention module, defaults to `MultiheadAttention`.
        p_dropout: Dropout rate for input `y`.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    scorer: nn.Module = DotScorer()
    num_heads: int = 4
    p_dropout: float = 0.3

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d = x.shape[-1]
        ctx, attn = MultiheadAttention(self.scorer, self.num_heads, self.p_dropout)(
            x, x, x, valid_lens, training
        )
        y = AddNorm(self.p_dropout)(x, ctx, training)
        ctx = nn.Sequential([nn.Dense(d), nn.relu, nn.Dense(d)])(y)
        return AddNorm(self.p_dropout)(y, ctx, training)


class TransformerEncoder(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        embedder: Embedding module for input points.
        scorer: Scoring module used to calculate query-key attention.
        num_heads: Number of attention heads per encoder block.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate for output.

    Returns:
        Input transformed by the encoder.
    """

    embedder: nn.Module = FixedSinusoidalEmbedding()
    scorer: nn.Module = DotScorer()
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
        x = self.embedder(x)
        for _ in range(self.num_blks):
            x = TransformerEncoderBlock(
                self.scorer.copy(), self.num_heads, self.p_dropout
            )(x, valid_lens, training)
        return x


class TransformerDecoderBlock(nn.Module):
    """A single decoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        scorer: Scoring module used to calculate query-key attention.
        p_dropout: Dropout rate for input `y`.

    Returns:
        Input transformed by a single decoder block.
    """

    scorer: nn.Module = DotScorer()
    p_dropout: float = 0.3

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        # TODO(danj): implement (may not need?)
        pass
