"""
Transformer architecture inspired by [d2l](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)'s version.
"""

from typing import Optional

import flax.linen as nn
import jax

from .attention import MultiheadAttention
from .fast_attention import MultiheadFastAttention
from .mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """A single encoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745).

    Args:
        attn: An attention module.
        norm: A normalization module.
        ffn: A feedforward network module.
        p_dropout: Dropout rate for residual connections.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    attn: nn.Module = MultiheadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64], nn.relu)
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        x_1 = self.norm(x)
        x_2, attn = self.attn(x_1, x_1, x_1, valid_lens, training, **kwargs)
        x_3 = x + drop(x_2)
        x_4 = self.norm.copy()(x_3)
        x_5 = self.ffn(x_4)
        return x_3 + drop(x_5), attn


class TransformerEncoder(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745).

    Args:
        num_blks: The number of blocks to use.
        num_reps: Number of times to repeat each block.
        blk: An encoder block.

    Returns:
        Input transformed by the encoder.
    """

    num_blks: int = 6
    num_reps: int = 1
    blk: nn.Module = TransformerEncoderBlock()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                x, _ = blk(x, valid_lens, training, **kwargs)
        return nn.LayerNorm()(x)


class TransformerDecoderBlock(nn.Module):
    """A single decoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).


    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745).

    .. note::
        This doesn't incorporate any logic for generative decoding one step at
        a time.

    Args:
        attn: An attention module.
        norm: A normalization module.
        ffn: A feedforward network module.
        p_dropout: Dropout rate for residual connections.

    Returns:
        Input transformed by a single decoder block.
    """

    attn: nn.Module = MultiheadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64], nn.relu)
    p_dropout: float = 0.0

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
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        x_dec_1 = self.norm(x_dec)
        x_dec_2, self_attn = self.attn(
            x_dec_1,
            x_dec_1,
            x_dec_1,
            valid_lens_dec,
            training,
            **kwargs,
        )
        x_dec_3 = x_dec + drop(x_dec_2)
        x_dec_4 = self.norm.copy()(x_dec_3)
        x_dec_5, cross_attn = self.attn.copy(name="cross_attn")(
            x_dec_4,
            x_enc,
            x_enc,
            valid_lens_enc,
            training,
            **kwargs,
        )
        x_dec_6 = x_dec_3 + drop(x_dec_5)
        x_dec_7 = self.ffn(x_dec_6)
        x_dec_8 = x_dec_6 + drop(x_dec_7)
        return x_dec_8, self_attn, cross_attn


class TransformerDecoder(nn.Module):
    """A transformer decoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745).

    Args:
        num_blks: Number of decoder blocks.
        num_reps: Number of times to repeat each block.
        blk: A decoder block.

    Returns:
        Input transformed by the decoder.
    """

    num_blks: int = 6
    num_reps: int = 1
    blk: nn.Module = TransformerDecoderBlock()
    ffn: nn.Module = MLP([128, 64], nn.relu)

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
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                x_dec, _, _ = blk(
                    x_dec,
                    x_enc,
                    valid_lens_dec,
                    valid_lens_enc,
                    training,
                    **kwargs,
                )
        return nn.LayerNorm()(x_dec)


class KRBlock(nn.Module):
    """A Kernel Regression Block.

    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745).

    Args:
        attn: An attention module.
        norm: A normalization module.
        ffn: A feedforward module.
        p_dropout: Dropout rate for residual connections.

    Returns:
        An instance of the `KRBlock` model.
    """

    attn: nn.Module = MultiheadFastAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64], nn.relu)
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        valid_lens: Optional[jax.Array] = None,
        training: bool = False,
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        qvs_1, kvs_1 = self.norm(qvs), self.norm(kvs)
        qvs_2, _ = self.attn(qvs_1, kvs_1, kvs_1, valid_lens, training)
        kvs_2, _ = self.attn(kvs_1, kvs_1, kvs_1, valid_lens, training)
        qvs_3, kvs_3 = qvs + drop(qvs_2), kvs + drop(kvs_2)
        norm_2 = self.norm.copy()
        qvs_4, kvs_4 = norm_2(qvs_3), norm_2(kvs_3)
        qvs_5, kvs_5 = self.ffn(qvs_4, training), self.ffn(kvs_4, training)
        return qvs_3 + drop(qvs_5), kvs_3 + drop(kvs_5)


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
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                qvs, kvs = blk(qvs, kvs, valid_lens, training)
        norm = nn.LayerNorm()
        return norm(qvs), norm(kvs)
