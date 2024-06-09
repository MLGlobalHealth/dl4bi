"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax

from .attention import Attention, MultiheadAttention


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
        attn: Attention module, defaults to `MultiheadAttention`.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x`.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    attn: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        d = x.shape[-1]
        d_ffn = self.d_ffn or 2 * d
        ctx, attn = self.attn(x, x, x, valid_lens, training, **kwargs)
        y = AddNorm(self.p_dropout)(x, ctx, training)
        ctx = nn.Sequential([nn.Dense(d_ffn), self.act_fn, nn.Dense(d)])(y)
        return AddNorm(self.p_dropout)(y, ctx, training), attn


class TransformerEncoder(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attn: Attention module to use.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x`.

    Returns:
        Input transformed by the encoder.
    """

    attn: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        d_ffn = self.d_ffn or 2 * x.shape[-1]
        x, _ = TransformerEncoderBlock(self.attn, self.p_dropout, d_ffn, self.act_fn)(
            x, valid_lens, training, **kwargs
        )
        for i in range(1, self.num_blks):
            x, _ = TransformerEncoderBlock(
                self.attn.copy(name=f"attn_{i}"),
                self.p_dropout,
                d_ffn,
                self.act_fn,
            )(x, valid_lens, training, **kwargs)
        return x


class TransformerDecoderBlock(nn.Module):
    """A single decoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This doesn't incorporate any logic for generative decoding one step at
        a time.

    Args:
        attn: Attention module to use.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x_dec`.

    Returns:
        Input transformed by a single decoder block.
    """

    attn: nn.Module = MultiheadAttention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

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
        d_ffn = self.d_ffn or 2 * d
        x_dec_2, attn_dec = self.attn(
            x_dec, x_dec, x_dec, valid_lens_dec, training, **kwargs
        )
        y_dec = AddNorm(self.p_dropout)(x_dec, x_dec_2, training)
        y_dec_enc, attn_enc = self.attn.copy(name="enc_attn")(
            y_dec, x_enc, x_enc, valid_lens_enc, training, **kwargs
        )
        z_dec_enc = AddNorm(self.p_dropout)(y_dec, y_dec_enc, training)
        z_dec_enc_2 = nn.Sequential([nn.Dense(d_ffn), self.act_fn, nn.Dense(d)])(
            z_dec_enc
        )
        out = AddNorm(self.p_dropout)(z_dec_enc, z_dec_enc_2, training)
        return out, attn_dec, attn_enc


class TransformerDecoder(nn.Module):
    """A transformer decoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        attn: Attention module to use.
        num_blks: Number of encoder blocks.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `x_dec`.

    Returns:
        Input transformed by the encoder.
    """

    attn: nn.Module = MultiheadAttention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

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
        d_ffn = self.d_ffn or 2 * d
        x_dec, _, _ = TransformerDecoderBlock(
            self.attn, self.p_dropout, d_ffn, self.act_fn
        )(x_dec, x_enc, valid_lens_dec, valid_lens_enc, training, **kwargs)
        for i in range(1, self.num_blks):
            x_dec, _, _ = TransformerDecoderBlock(
                self.attn.copy(name=f"attn_{i}"),
                self.p_dropout,
                d_ffn,
                self.act_fn,
            )(x_dec, x_enc, valid_lens_dec, valid_lens_enc, training, **kwargs)
        return nn.Dense(d)(x_dec)


class KRBlock(nn.Module):
    """A Kernel Regression Block.

    Args:
        attn: An attention module.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `kvs`.
        act_fn: Activation function, defaults to relu.

    Returns:
        An instance of the `KRBlock` model.
    """

    attn: nn.Module = Attention()
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d = kvs.shape[-1]
        d_ffn = self.d_ffn or 2 * d
        add_norm = AddNorm(self.p_dropout)
        ffn = nn.Sequential([nn.Dense(d_ffn), self.act_fn, nn.Dense(d)])
        qvs2, _ = self.attn(qvs, kvs, kvs, valid_lens)
        kvs2, _ = self.attn(kvs, kvs, kvs, valid_lens)
        qvs3, kvs3 = add_norm(qvs, qvs2), add_norm(kvs, kvs2)
        qvs4, kvs4 = ffn(qvs3), ffn(kvs3)
        return add_norm(qvs3, qvs4), add_norm(kvs3, kvs4)


class KRStack(nn.Module):
    """A stack of `KRBlock`s.

    Args:
        attn: An attention module.
        num_blks: Number of `KRBlock`s.
        p_dropout: Dropout rate `AddNorm`s.
        d_ffn: Optional dim for feed forward, defaults to twice the last
            dimension of input `kvs`.

    Returns:
        An instance of a `KRStack`.
    """

    attn: nn.Module = Attention()
    num_blks: int = 3
    p_dropout: float = 0.0
    d_ffn: Optional[int] = None
    act_fn: Callable = nn.relu
    skip_every_n: Optional[int] = None

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        d_ffn = self.d_ffn or 2 * kvs.shape[-1]
        add_norm = AddNorm(self.p_dropout)
        qvs, kvs = KRBlock(
            self.attn,
            self.p_dropout,
            d_ffn,
            self.act_fn,
        )(qvs, kvs, valid_lens, training)
        skip_qvs, skip_kvs = qvs, kvs
        for i in range(1, self.num_blks):
            is_skip = self.skip_every_n and i % self.skip_every_n == 0
            if is_skip:
                qvs = add_norm(skip_qvs, qvs)
                kvs = add_norm(skip_kvs, kvs)
            qvs, kvs = KRBlock(
                self.attn.copy(name=f"attn_{i}"),
                self.p_dropout,
                d_ffn,
                self.act_fn,
            )(qvs, kvs, valid_lens, training)
            if is_skip:
                skip_qvs = qvs
                skip_kvs = kvs
        return qvs, kvs
