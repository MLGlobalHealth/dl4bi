"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from typing import Optional

import flax.linen as nn
import jax

from .attention import MultiheadAttention


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
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x`.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    attention: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        d = x.shape[-1]
        d_ffn = self.d_ffn or 2 * x.shape[-1]
        ctx, attn = self.attention(x, x, x, valid_lens, training, **kwargs)
        y = AddNorm(self.p_dropout)(x, ctx, training)
        ctx = nn.Sequential([nn.Dense(d_ffn), nn.relu, nn.Dense(d)])(y)
        return AddNorm(self.p_dropout)(y, ctx, training), attn


class TransformerEncoder(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attention: Attention module to use.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x`.

    Returns:
        Input transformed by the encoder.
    """

    attention: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        d_ffn = self.d_ffn or 2 * x.shape[-1]
        x, _ = TransformerEncoderBlock(self.attention, self.p_dropout, d_ffn)(
            x, valid_lens, training, **kwargs
        )
        for i in range(1, self.num_blks):
            x, _ = TransformerEncoderBlock(
                self.attention.copy(name=f"attention_{i}"), self.p_dropout, d_ffn
            )(x, valid_lens, training, **kwargs)
        return x


class TransformerDecoderBlock(nn.Module):
    """A single decoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This doesn't incorporate any logic for generative decoding one step at
        a time.

    Args:
        attention: Attention module to use.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x_dec`.

    Returns:
        Input transformed by a single decoder block.
    """

    attention: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x_dec: jax.Array,
        x_enc: jax.Array,
        valid_lens_dec: Optional[jax.Array] = None,
        valid_lens_enc: Optional[jax.Array] = None,
        training=False,
        **kwargs,
    ):
        d = x_dec.shape[-1]
        d_ffn = self.d_ffn or 2 * x_dec.shape[-1]
        x_dec_2, attn_dec = self.attention(
            x_dec, x_dec, x_dec, valid_lens_dec, training, **kwargs
        )
        y_dec = AddNorm(self.p_dropout)(x_dec, x_dec_2, training)
        y_dec_enc, attn_enc = self.attention.copy(name="enc_attention")(
            y_dec, x_enc, x_enc, valid_lens_enc, training, **kwargs
        )
        z_dec_enc = AddNorm(self.p_dropout)(y_dec, y_dec_enc, training)
        z_dec_enc_2 = nn.Sequential([nn.Dense(d_ffn), nn.relu, nn.Dense(d)])(z_dec_enc)
        out = AddNorm(self.p_dropout)(z_dec_enc, z_dec_enc_2, training)
        return out, attn_dec, attn_enc


class TransformerDecoder(nn.Module):
    """A transformer decoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attention: Attention module to use.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x_dec`.

    Returns:
        Input transformed by the encoder.
    """

    attention: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        x_dec: jax.Array,
        x_enc: jax.Array,
        valid_lens_dec: Optional[jax.Array] = None,
        valid_lens_enc: Optional[jax.Array] = None,
        training=False,
        **kwargs,
    ):
        d = x_dec.shape[-1]
        d_ffn = self.d_ffn or 2 * x_dec.shape[-1]
        x_dec, _, _ = TransformerDecoderBlock(self.attention, self.p_dropout, d_ffn)(
            x_dec, x_enc, valid_lens_dec, valid_lens_enc, training, **kwargs
        )
        for i in range(1, self.num_blks):
            x_dec, _, _ = TransformerDecoderBlock(
                self.attention.copy(name=f"attention_{i}"), self.p_dropout, d_ffn
            )(x_dec, x_enc, valid_lens_dec, valid_lens_enc, training, **kwargs)
        return nn.Dense(d)(x_dec)
