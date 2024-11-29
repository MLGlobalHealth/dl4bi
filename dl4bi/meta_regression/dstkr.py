from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp

from ..core import (
    MLP,
    KRBlock,
    MultiHeadAttention,
    SpatioTemporalMLPAttention,
)


class DSTKR(nn.Module):
    """Deep Spatiotemporal Kernel Regression.

    Args:

    Returns:
        An instance of the `DSTKR` model.
    """

    num_blks: int = 6
    num_reps: int = 1
    min_std: float = 0.0
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    max_dist: float = 1
    attn: nn.Module = MultiHeadAttention(SpatioTemporalMLPAttention())
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
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        qvs, kvs = self.norm(self.embed_all(test)), self.norm(self.embed_all(ctx))
        vnode = self.param("vnode", init.ones(), (kvs.shape[-1],))
        qk_kwargs = {"qs_s": s_test, "ks_s": s_ctx, "vnode": vnode}
        kk_kwargs = {"qs_s": s_ctx, "ks_s": s_ctx, "vnode": vnode}
        for _ in range(self.num_blks):
            attn, ffn = self.attn.copy(), self.ffn.copy()
            for _ in range(self.num_reps):
                norm = self.norm.copy()
                blk = KRBlock(attn, norm, ffn)
                qvs, kvs = blk(qvs, kvs, valid_lens_ctx, training, qk_kwargs, kk_kwargs)
                # TODO(danj): update vnode from kvs
                # vnode = update
        qvs = self.norm.copy()(qvs)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        # TODO(danj): prediction head for vnode
        f_mu, f_std = 0, 0
        return f_mu, f_std
