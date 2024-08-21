from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from .mlp import MLP
from .utils import causal_mask, mask


class DotScorer(nn.Module):
    r"""Performs dot product attention scoring from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    $$a(\mathbf{q},\mathbf{k}) = \frac{\mathbf{q}^\intercal\mathbf{k}}{\sqrt{d}}$$
    """

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        d = qs.shape[-1]
        return jnp.einsum("bqd,bkd->bqk", qs, ks) / jnp.sqrt(d)


class AdditiveScorer(nn.Module):
    r"""Performs additive attention scoring from ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473).

    $$a(\mathbf{q},\mathbf{k}) = \mathbf{w}_v^\intercal(\mathbf{W}_q\mathbf{q}+\mathbf{W}_k\mathbf{k})$$
    """

    num_hidden: int = 128

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        qs = nn.Dense(self.num_hidden, use_bias=False)(qs)
        ks = nn.Dense(self.num_hidden, use_bias=False)(ks)
        # [B, Q, 1, H] + [B, 1, K, H]
        feats = jnp.expand_dims(qs, axis=2) + jnp.expand_dims(ks, axis=1)
        feats = nn.tanh(feats)
        return nn.Dense(1, use_bias=False)(feats).squeeze(-1)


class MultiplicativeScorer(nn.Module):
    r"""Performs multiplicative attention scoring from ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025).

    $$a(\mathbf{q},\mathbf{k}) = \mathbf{q}^\intercal\mathbf{W}_a\mathbf{k}$$
    """

    @nn.compact
    def __call__(self, qs: jax.Array, ks: jax.Array):
        qs = nn.Dense(qs.shape[-1], use_bias=False)(qs)
        return DotScorer()(qs, ks)


class Attention(nn.Module):
    r"""Performs (masked) query-key-value attention with dropout.

    Args:
        scorer: A module used to provide similarity scores between queries and keys.
        p_dropout: A dropout rate.
        is_causal: A boolean indicating whether attention is causal. If true,
            `valid_lens` is ignored and a causal mask is applied.

    Returns:
        An `Attention` module.

    .. note:: This assumes all queries, keys, and values are already embedded, i.e.
        $$
        \begin{aligned}
            \mathbf{Q}&=\mathbf{W}^Q\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{K}&=\mathbf{W}^K\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{V}&=\mathbf{W}^V\mathbf{Y}\in\mathbb{R}^{N\times D_V} \\\\
        \end{aligned}
        $$
    """

    scorer: nn.Module = DotScorer()
    p_dropout: float = 0.0
    is_causal: bool = False

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_Q]
        ks: jax.Array,  # [B, K, D_K]
        vs: jax.Array,  # [B, V, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, Q]
        training: bool = False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        drop = nn.Dropout(self.p_dropout, deterministic=not training)
        scores = self.scorer(qs, ks)
        if self.is_causal:
            scores = causal_mask(scores)
        elif valid_lens is not None:
            scores = mask(scores, valid_lens)
        attn = nn.softmax(scores, axis=-1)
        ctx = drop(attn) @ vs
        return ctx, attn


class MultiheadAttention(nn.Module):
    r"""Performs multihead (masked) query-key-value attention with dropout.

    Args:
        num_heads: Number of heads for attention module.
        p_dropout: A dropout rate for attention.
        is_causal: A boolean indicating whether attention is causal. If true,
            `valid_lens` is ignored and a causal mask is applied.

    Returns:
        A `MultiheadAttention` module.

    .. note:: This assumes all queries, keys, and values are already embedded, i.e.
        $$
        \begin{aligned}
            \mathbf{Q}&=\mathbf{W}^Q\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{K}&=\mathbf{W}^K\mathbf{X}\in\mathbb{R}^{N\times D_{Q,K}} \\\\
            \mathbf{V}&=\mathbf{W}^V\mathbf{Y}\in\mathbb{R}^{N\times D_V} \\\\
        \end{aligned}
        $$
    """

    proj_qs: nn.Module = MLP([64])
    proj_ks: nn.Module = MLP([64])
    proj_vs: nn.Module = MLP([64])
    proj_out: nn.Module = MLP([64])
    scorer: nn.Module = DotScorer()
    num_heads: int = 4
    p_dropout: float = 0.0
    is_causal: bool = False

    @nn.compact
    def __call__(
        self,
        qs: jax.Array,  # [B, Q, D_QK]
        ks: jax.Array,  # [B, K, D_QK]
        vs: jax.Array,  # [B, V, D_V]
        valid_lens: Optional[jax.Array] = None,  # [B] or [B, K]
        training=False,
        **kwargs,
    ):
        r"""Performs forward pass of network.

        Args:
            qs: Queries of dimension $\mathbb{R}^{B\times Q\times D_{Q,K}}$
            ks: Keys of dimension $\mathbb{R}^{B\times K\times D_{Q,K}}$
            vs: Values of dimension $\mathbb{R}^{B\times K\tiems D_V}$
            valid_lens: Mask consisting of valid length per sequence of dimension
                $\mathbb{R}^B$ or $\mathbb{R}^{B\times K}$
            training: Boolean indicating whether currently training.

        Returns:
            `ctx` and `attn`, the updated values and attention weights.
        """
        qs, ks, vs = self.proj_qs(qs), self.proj_ks(ks), self.proj_vs(vs)
        (B, Q, D_QK), K, D_V, H = qs.shape, ks.shape[1], vs.shape[-1], self.num_heads
        D_QK_H, D_V_H = D_QK // H, D_V // H
        # [B, {Q,K}, D_{QK,V}] -> [B * H, {Q,K}, D_{QK,V}_H]
        qs = qs.reshape(B, Q, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, Q, D_QK_H)
        ks = ks.reshape(B, K, H, D_QK_H).transpose(0, 2, 1, 3).reshape(-1, K, D_QK_H)
        vs = vs.reshape(B, K, H, D_V_H).transpose(0, 2, 1, 3).reshape(-1, K, D_V_H)
        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, H, axis=0)
        # [B * H, Q, D_V_H], [B * H, Q, K]
        ctx, attn = Attention(self.scorer, self.p_dropout, self.is_causal)(
            qs, ks, vs, valid_lens, training
        )
        # [B * H, Q, D_V_H] -> [B, Q, D_V]
        ctx = ctx.reshape(B, H, Q, D_V_H).transpose(0, 2, 1, 3).reshape(B, Q, D_V)
        return self.proj_out(ctx), attn.reshape(B, H, Q, K)
