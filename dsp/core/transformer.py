"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax

from .attention import MultiheadAttention
from .fast_attention import MultiheadFastAttention
from .mlp import MLP


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

    This uses pre-normalization as specified in https://arxiv.org/pdf/2002.04745.

    Args:
        attn: An attention module (MultiheadFastAttention by default).
        norm: A normalization module (LayerNorm by default).
        ffn: A feedforward module.

    Returns:
        An instance of the `KRBlock` model.
    """

    attn: nn.Module = MultiheadFastAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64], nn.relu)

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        qvs_1, kvs_1 = self.norm(qvs), self.norm(kvs)
        qvs_2, _ = self.attn(qvs_1, kvs_1, kvs_1, valid_lens, training)
        kvs_2, _ = self.attn(kvs_1, kvs_1, kvs_1, valid_lens, training)
        qvs_3, kvs_3 = qvs + qvs_2, kvs + kvs_2
        qvs_4, kvs_4 = self.norm(qvs_3), self.norm(kvs_3)
        qvs_5, kvs_5 = self.ffn(qvs_4, training), self.ffn(kvs_4, training)
        return qvs_3 + qvs_5, kvs_3 + kvs_5


class KRStack(nn.Module):
    """A stack of `KRBlock`s.

    Args:
        num_blks: Number of blocks to use.
        num_reps: Number of times to repeat each block.
        blk: An instance of the block module.

    Returns:
        An instance of a `KRStack`.
    """

    num_blks: int = 6
    num_reps: int = 1
    blk: nn.Module = KRBlock()

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        layer_norm = nn.LayerNorm()
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                qvs, kvs = blk(qvs, kvs, valid_lens, training)
        return layer_norm(qvs), layer_norm(kvs)
