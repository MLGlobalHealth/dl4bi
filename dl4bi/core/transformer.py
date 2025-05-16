from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers as init
from jax import jit

from .attention import MultiHeadAttention, TEMultiHeadAttention
from .mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """A single encoder block from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    .. note::
        This formulation uses [pre-normalization](https://arxiv.org/pdf/2002.04745) by default.

    Args:
        attn: An attention module.
        norm: A normalization module.
        ffn: A feedforward network module.
        p_dropout: Dropout rate for residual connections.
        pre_norm: Boolean indicating whether to use-prenormalization.

    Returns:
        Input transformed by a single self-attention encoder block.
    """

    attn: nn.Module = MultiHeadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64], nn.relu)
    p_dropout: float = 0.0
    pre_norm: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        if self.pre_norm:
            x_1 = self.norm(x)
            x_2, *rest = self.attn(x_1, x_1, x_1, mask, training, **kwargs)
            x_3 = x + drop(x_2)
            x_4 = self.norm.copy()(x_3)
            x_5 = self.ffn(x_4)
            return x_3 + drop(x_5), *rest
        # post-norm, original formulation
        x_1, *rest = self.attn(x, x, x, mask, training, **kwargs)
        x_2 = self.norm(x + drop(x_1))
        x_3 = self.ffn(x_2)
        x_4 = self.norm.copy()(x_2 + drop(x_3))
        return x_4, *rest


class TransformerEncoder(nn.Module):
    """A transformer encoder inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    Args:
        num_blks: The number of blocks to use.
        blk: An encoder block.
        norm: Final normalization module used before output.

    Returns:
        Input transformed by the encoder.
    """

    num_blks: int = 6
    blk: nn.Module = TransformerEncoderBlock()
    norm: nn.Module = nn.LayerNorm()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        for _ in range(self.num_blks):
            x, *_ = self.blk.copy()(x, mask, training, **kwargs)
        if self.blk.pre_norm:
            x = self.norm(x)
        return x


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

    attn: nn.Module = MultiHeadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 64])
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x_dec: jax.Array,
        x_enc: jax.Array,
        mask_dec: Optional[jax.Array] = None,
        mask_enc: Optional[jax.Array] = None,
        training=False,
        qq_kwargs={},
        qk_kwargs={},
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        x_dec_1, x_enc_1 = self.norm(x_dec), self.norm(x_enc)
        x_dec_2, self_attn = self.attn(
            x_dec_1,
            x_dec_1,
            x_dec_1,
            mask_dec,
            training,
            **qq_kwargs,
        )
        x_dec_3 = x_dec + drop(x_dec_2)
        x_dec_4 = self.norm.copy()(x_dec_3)
        x_dec_5, cross_attn = self.attn.copy(name="cross_attn")(
            x_dec_4,
            x_enc_1,
            x_enc_1,
            mask_enc,
            training,
            **qk_kwargs,
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
        blk: A decoder block.

    Returns:
        Input transformed by the decoder.
    """

    num_blks: int = 6
    blk: nn.Module = TransformerDecoderBlock()
    ffn: nn.Module = MLP([128, 64], nn.relu)

    @nn.compact
    def __call__(
        self,
        x_dec: jax.Array,
        x_enc: jax.Array,
        mask_dec: Optional[jax.Array] = None,
        mask_enc: Optional[jax.Array] = None,
        training=False,
        **kwargs,
    ):
        for _ in range(self.num_blks):
            x_dec, _, _ = self.blk.copy()(
                x_dec,
                x_enc,
                mask_dec,
                mask_enc,
                training,
                **kwargs,
            )
        return nn.LayerNorm()(x_dec)


class TEBlock(nn.Module):
    """TEBlock from [Translation Equivariant Transformer Netural Processes](https://arxiv.org/abs/2406.12409).

    .. note::
        This actually corresponds to a `MultiHeadCrossTEAttentionLayer` in the original code.
    """

    attn: nn.Module = TEMultiHeadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([128, 128], nn.relu)
    p_dropout: float = 0.0
    pre_norm: bool = True

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,
        ks: jax.Array,
        qs_s: jax.Array,
        ks_s: jax.Array,
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        if self.pre_norm:
            qs_1, ks_1 = self.norm(qs), self.norm(ks)
            qs_2, qs_s, *rest = self.attn(
                qs_1, ks_1, ks_1, qs_s, ks_s, mask, training, **kwargs
            )
            qs_3 = qs + drop(qs_2)
            qs_4 = self.norm.copy()(qs_3)
            qs_5 = self.ffn(qs_4)
            return qs_3 + drop(qs_5), qs_s, *rest
        # post-norm, original formulation
        qs_1, qs_s, *rest = self.attn(qs, ks, ks, qs_s, ks_s, mask, training, **kwargs)
        qs_2 = self.norm(qs + drop(qs_1))
        qs_3 = self.ffn(qs_2)
        qs_4 = self.norm.copy()(qs_2 + drop(qs_3))
        return qs_4, qs_s, *rest


class TEISTEncoder(nn.Module):
    """TEISTEncoder from [Translation Equivariant Transformer Netural Processes](https://arxiv.org/abs/2406.12409)."""

    num_blks: int = 5
    num_latents: int = 128
    embed_dim: int = 128
    ps_to_ks_blk: nn.Module = TEBlock()
    ks_to_ps_blk: nn.Module = TEBlock()
    qs_to_ps_blk: nn.Module = TEBlock()

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_q]
        ks: jax.Array,  # [B, K, D_k]
        qs_s: jax.Array,  # [B, Q, D_s]
        ks_s: jax.Array,  # [B, K, D_s]
        mask: Optional[jax.Array] = None,
        training: bool = False,
        **kwargs,
    ):
        (B, _L, D_s), Z, E = qs_s.shape, self.num_latents, self.embed_dim
        batchify = jit(lambda v: jnp.repeat(v, B, axis=0))
        ps = self.param("pseudo_tokens", init.normal(stddev=1.0), (1, Z, E))
        ps_s = self.param("pseudo_locs", init.normal(stddev=1.0), (1, Z, D_s))
        ps, ps_s = batchify(ps), batchify(ps_s)
        # shift ps_s to mean ks_s location
        if mask is None:
            ps_s += ks_s.mean(axis=1, keepdims=True)
        else:
            ps_s += ks_s.mean(axis=1, keepdims=True, where=mask[..., None])
        for _ in range(self.num_blks):
            ps, ps_s, _ = self.ps_to_ks_blk.copy()(
                ps, ks, ps_s, ks_s, mask, training, **kwargs
            )
            ks, ks_s, _ = self.ks_to_ps_blk.copy()(
                ks, ps, ks_s, ps_s, None, training, **kwargs
            )
            qs, qs_s, _ = self.qs_to_ps_blk.copy()(
                qs, ps, qs_s, ps_s, None, training, **kwargs
            )
        return qs


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

    attn: nn.Module = MultiHeadAttention()
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([256, 64])
    p_dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        qvs: jax.Array,
        kvs: jax.Array,
        mask: Optional[jax.Array] = None,
        training: bool = False,
        qk_kwargs: dict = {},
        kk_kwargs: dict = {},
        **kwargs,
    ):
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        qvs_1, kvs_1 = self.norm(qvs), self.norm(kvs)
        qvs_2, _ = self.attn(qvs_1, kvs_1, kvs_1, mask, training, **qk_kwargs)
        kvs_2, _ = self.attn(kvs_1, kvs_1, kvs_1, mask, training, **kk_kwargs)
        qvs_3, kvs_3 = qvs + drop(qvs_2), kvs + drop(kvs_2)
        norm_2 = self.norm.copy()
        qvs_4, kvs_4 = norm_2(qvs_3), norm_2(kvs_3)
        qvs_5, kvs_5 = self.ffn(qvs_4, training), self.ffn(kvs_4, training)
        return qvs_3 + drop(qvs_5), kvs_3 + drop(kvs_5)
