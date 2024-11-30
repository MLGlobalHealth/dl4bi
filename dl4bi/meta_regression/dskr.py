from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from einops import repeat
from jax import jit
from jax.tree_util import Partial
from jraph import GraphsTuple
from scipy.spatial import KDTree
from sps.kernels import l2_dist

from dl4bi.core.attention import MultiHeadAttention

from ..core import (
    MLP,
    KRBlock,
    SpatioTemporalMLPAttention,
)

# Embed x, s, t
# Create a vnode
# Build graph
# Convolve
#  qs-ks
#  ks-ks
#  vnode-ks
# prediction head
#  vnode
#  qs
# Can we condition projections for attention on global state?


def custom_k_nearest_senders(
    rx: jax.Array,
    tx: jax.Array,
    k: int,
    dist: Callable = l2_dist,
):
    """Retrieves k-nearest senders, but uses a $O(n^2)$ memory."""
    d = dist(rx, tx)
    idx = jnp.argsort(d, axis=-1)
    d = jnp.take_along_axis(d, idx, axis=-1)
    return idx[:, :k].flatten(), d[:, :k].flatten()


def k_nearest_senders(rx: jax.Array, tx: jax.Array, k: int):
    d, idx = KDTree(tx).query(rx, k)
    return idx.flatten(), d.flatten()


class GDSKR(nn.Module):
    """GDSKR

    .. note::
        Fixed effects can be embedded with `embed_s`, i.e. if the "index"
        consists of [fixed effects, space, time], `embed_s` could be a Flax
        module that embeds them separately and concatenates the output.

    .. note::
        When the index set, `s`, includes fixed effects or features that
        do not factor into calculating the k-nearest neighbors, you
        can override `k_nearest_senders`.

    .. warning::
        `min(valid_lens_ctx)` and `min(valid_lens_test)` must both
        be greater than `k`.
    """

    k: int = 10
    k_nearest_senders: Callable = k_nearest_senders
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)

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
        (B, Q), K = s_test.shape[:-1], s_ctx.shape[1]
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(K, B)
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(Q, B)
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        # calculate node features
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        x_test, x_ctx = self.norm(self.embed_all(test)), self.norm(self.embed_all(ctx))
        # build localized graphs
        graphs = []
        receivers = jit(lambda n: jnp.repeat(jnp.arange(n), self.k))
        for b in range(B):
            n_c, n_t = valid_lens_ctx[b], valid_lens_test[b]
            s_c, s_t = s_ctx[b, :n_c], s_test[b, :n_t]
            x_c, x_t = x_ctx[b, :n_c], x_test[b, :n_t]
            tx_cc, d_cc = self.k_nearest_senders(s_c, s_c, self.k)
            tx_ct, d_ct = self.k_nearest_senders(s_t, s_c, self.k)
            rx_cc, rx_ct = receivers(s_c), receivers(s_t)
            g = GraphsTuple(
                nodes=stack(x_c, x_t),
                edges=stack(d_cc, d_ct),
                senders=stack(tx_cc, tx_ct),
                receivers=stack(rx_cc, n_c + rx_ct),
                n_node=n_c + n_t,
                n_edge=(n_c + n_t) * self.k,
                globals=jnp.max(x_c, axis=0, keepdims=True),
            )
            graphs += [g]


class DSKR(nn.Module):
    """Deep Sparse Kernel Regression.

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
    max_dist: float = float("inf")
    attn: nn.Module = SpatioTemporalMLPAttention()
    vattn: nn.Module = MultiHeadAttention()
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
        vnode = self.param("vnode", init.ones, (kvs.shape[-1],))
        vnode = repeat(vnode, "D -> B 1 D", B=qvs.shape[0])
        qk_kwargs = {"qs_s": s_test, "ks_s": s_ctx, "vnode": vnode}
        kk_kwargs = {"qs_s": s_ctx, "ks_s": s_ctx, "vnode": vnode}
        for _ in range(self.num_blks):
            attn, ffn, vattn = self.attn.copy(), self.ffn.copy(), self.vattn.copy()
            for _ in range(self.num_reps):
                norm = self.norm.copy()
                # TODO(danj): add vnode to kvs?
                vnode, _ = vattn(vnode, kvs, kvs, valid_lens_ctx, training)
                blk = KRBlock(attn, norm, ffn)
                qvs, kvs = blk(qvs, kvs, valid_lens_ctx, training, qk_kwargs, kk_kwargs)
        qvs = self.norm.copy()(qvs)
        f_dist = self.head(qvs, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        # TODO(danj): prediction head for vnode
        return f_mu, f_std
