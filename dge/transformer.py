"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import DotScorer, MultiheadAttention


class AddNorm(nn.Module):
    """Performs add and norm from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        p_dropout: Dropout rate for input `y`.

    Returns:
        Add-and-normed input.
    """

    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, training: bool = False):
        y = nn.Dropout(self.p_dropout, deterministic=not training)(y)
        return nn.LayerNorm()(x + y)


class TransformerEncoderBlock(nn.Module):
    """A single encoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attention: Attention module, defaults to `MultiheadAttention`.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    attention: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0

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
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attention: Attention module to use.
        num_heads: Number of attention heads per encoder block.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate for output.

    Returns:
        Input transformed by the encoder.
    """

    attention: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        x = TransformerEncoderBlock(self.attention, self.p_dropout)(
            x, valid_lens, training
        )
        for i in range(1, self.num_blks):
            x = TransformerEncoderBlock(
                self.attention.copy(name=f"attention_{i}"), self.p_dropout
            )(x, valid_lens, training)
        return x


# TODO(danj): test implementation
class TransformerDecoderBlock(nn.Module):
    """A single decoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        i: Identity of block.
        attention: Attention module to use.

    Returns:
        Input transformed by a single decoder block.
    """

    i: int
    attention: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, state, training=False):
        d = x.shape[-1]
        enc_outputs, enc_valid_lens, vs = state[0], state[1], state[2][self.i]
        vs = x if vs is None else jnp.concatenate((vs, x), axis=1)
        state[2][self.i] = vs
        # TODO(danj): perhaps remove this for our case?
        if training:
            B, S_test, _ = x.shape
            dec_valid_lens = jnp.tile(jnp.arange(1, S_test + 1), (B, 1))
        else:
            dec_valid_lens = None
        x2, attn1 = self.attention(x, vs, vs, dec_valid_lens, training)
        y = AddNorm(self.p_dropout)(x, x2, training)
        y2, attn2 = self.attention.copy(name="enc_attention")(
            y, enc_outputs, enc_outputs, enc_valid_lens, training
        )
        z = AddNorm(self.p_dropout)(y, y2, training)
        z2 = nn.Sequential([nn.Dense(d), nn.relu, nn.Dense(d)])(z)
        return AddNorm(self.p_dropout)(z, z2, training), state, attn1, attn2


# TODO(danj): test implementation
class TransformerDecoder(nn.Module):
    attention: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, state, training=False):
        d = x.shape[-1]
        x, state, _, _ = TransformerDecoderBlock(0, self.attention, self.p_dropout)(
            x, state, training
        )
        for i in range(1, self.num_blks):
            x, state, _, _ = TransformerDecoderBlock(
                i, self.attention.copy(name=f"attention_{i}"), self.p_dropout
            )(x, state, training)
        return nn.Dense(d)(x), state
