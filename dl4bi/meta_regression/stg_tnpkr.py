from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap
from sps.kernels import l2_dist_sq, l2_dist

from ..core import MLP, DistanceBias, GraphDistanceBias, TemporalBias, FusedAttention, KRBlock, MultiHeadAttention
import os


class STGTNPKR(nn.Module):
    """Spatial Temporal Graph Transformer Neural Process - Kernel Regression (STG-TNP-KR).

    Args:
        num_blks: Number of `KRBlocks` to use.
        num_reps: Number of times to repeat each `KRBlock`.
        min_std: Minimum pointwise standard deviation.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        dist: A distance function used to calculate pairwise distances between
            two arrays.
        bias: A bias module that consumes pairwise distances.
        attn: An attention module used in `KRBlocks`.
        norm: A normalization module used in `KRBlocks`.
        ffn: A FeedForward Network used in `KRBlocks`.
        head: A prediction head.

    Returns:
        An instance of the `TNP-KR` model.
    """

    num_blks: int = 6
    num_reps: int = 1
    min_std: float = 0.0
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    dist: Callable = l2_dist
    graph_bias_flag: bool = True
    graph_bias: nn.Module = GraphDistanceBias()
    temporal_bias_flag: bool = True
    temporal_bias: nn.Module = TemporalBias()
    attn: nn.Module = MultiHeadAttention(FusedAttention())
    norm: nn.Module = nn.LayerNorm()
    ffn: nn.Module = MLP([256, 64], nn.gelu)
    head: nn.Module = MLP([256, 64, 2], nn.gelu)

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, D_S]
        f_ctx: jax.Array,  # [B, L_ctx, D_F]
        s_test: jax.Array,  # [B, L_test, D_S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        # inv_permute_idx: Optional[jax.Array] = None,
        **kwargs,
    ):
        r"""Run module forward.

        Args:
            rng: A psuedo-random number generator.
            s_ctx: An index set array of shape `[B, L_ctx, D_S]` where
                `B` is batch size, `L_ctx` is number of context
                locations, and `L_ctx` is the dimension of each location.
            f_ctx: A function value array of shape `[B, L_ctx, D_F]` where `B` is
                batch size, `L_ctx` is number of context locations, and `D_F` is
                the dimension of each function value.
            s_test: A location array of shape `[B, L_test, D_S]` where `B` is
                batch size, `L_test` is number of test locations, and `D_S`
                is the dimension of each location.
            valid_lens_ctx: An optional array of shape `(B,)` indicating the
                valid positions for each `L_ctx` sequence in the batch.
            valid_lens_test: An optional array of shape `(B,)` indicating the
                valid positions for each `L_test` sequence in the batch.
            training: A boolean indicating whether this call is performed during
                training.

        Returns:
            $\mu_f,\sigma_f\in\mathbb{R}^{B\times L_\text{test}\times D_F}$.
        """
        vdist = vmap(self.dist)
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        qvs, kvs = self.norm(self.embed_all(test)), self.norm(self.embed_all(ctx))
        
        # d_qk, d_kk = vdist(s_test, s_ctx), vdist(s_ctx, s_ctx)
        d_qk_temporal, d_kk_temporal = vdist(s_test[:,:,-1], s_ctx[:,:,-1]), vdist(s_ctx[:,:,-1], s_ctx[:,:,-1])
        # print('s_test[:,:,-1]:', s_test[:,:,-1])
        # print('s_ctx[:,:,-1]:', s_ctx[:,:,-1])
        # print('d_qk:', d_qk)
        # d_qk, d_kk = jnp.zeros_like(d_qk), jnp.zeros_like(d_kk) # ignore the distance
        
        graph_dist = kwargs['graph_dist']
        inv_permute_idx = kwargs['inv_permute_idx']
        inv_permute_idx_test = kwargs['inv_permute_idx_test']
    
        # TODO: double check inv permute is correct
        # TODO: update to suit general use (without batch)
        permute_idx = inv_permute_idx.argsort(axis=1) # B x L
        permute_idx_test = inv_permute_idx_test.argsort(axis=1) # B x L
        B = inv_permute_idx.shape[0]
        L_test = inv_permute_idx_test.shape[1]
        L_ctx = inv_permute_idx.shape[1]
        d_qk_graph = jnp.zeros((B, L_test, L_ctx))
        d_kk_graph = jnp.zeros((B, L_ctx, L_ctx))
        for b in range(inv_permute_idx.shape[0]):
            d_qk_graph = d_qk_graph.at[b].set(graph_dist[permute_idx_test[b]][:, permute_idx[b]])
            d_kk_graph = d_kk_graph.at[b].set(graph_dist[permute_idx[b]][:, permute_idx[b]])
        
        # d_qk_permuted = graph_dist[permute_idx_test][:, permute_idx]
        # d_kk_permuted = graph_dist[permute_idx][:, permute_idx]
        # d_qk_graph = jnp.repeat(d_qk_permuted[None,:,:], len(s_ctx), axis=0) # B x L x L
        # d_kk_graph = jnp.repeat(d_kk_permuted[None,:,:], len(s_ctx), axis=0) # B x L x L
        
        for _ in range(self.num_blks):
            attn, ffn = self.attn.copy(), self.ffn.copy()
            for _ in range(self.num_reps):
                graph_bias = self.graph_bias.copy()
                gb_qk, gb_kk = graph_bias(d_qk_graph), graph_bias(d_kk_graph)
                if not self.graph_bias_flag:
                    gb_qk, gb_kk = jnp.zeros_like(gb_qk), jnp.zeros_like(gb_kk)
                temporal_bias = self.temporal_bias.copy()
                tb_qk, tb_kk = temporal_bias(d_qk_temporal), temporal_bias(d_kk_temporal)
                if not self.temporal_bias_flag:
                    tb_qk, tb_kk = jnp.zeros_like(tb_qk), jnp.zeros_like(tb_kk)
                b_qk = gb_qk + tb_qk
                b_kk = gb_kk + tb_kk
                
                norm = self.norm.copy()
                blk = KRBlock(attn, norm, ffn)
                qvs, kvs = blk(qvs, kvs, b_qk, b_kk, valid_lens_ctx, training, inv_permute_idx)
        qvs = self.norm.copy()(qvs)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        return f_mu, f_std
