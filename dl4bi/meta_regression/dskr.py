from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import vmap
from jraph import GraphsTuple

from ..core import MLP, GraphKRBlock, TISABias, k_nearest_senders, mask_from_valid_lens


# TODO(danj): include global vnode conditioning
# TODO(danj): add masks for when k < num_ctx
class DSKR(nn.Module):
    """DSKR

    .. note::
        When the index set, `s`, includes fixed effects or features that
        do not factor into calculating the k-nearest neighbors, you
        can override `k_nearest_senders` and `embed_s`.

    .. warning::
        `min(valid_lens_ctx)` and `min(valid_lens_test)` must both
        be greater than `k`.
    """

    k: int = 10
    k_nearest_senders: Callable = k_nearest_senders
    num_blks: int = 6
    num_reps: int = 1
    min_std: float = 0.0
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    bias: nn.Module = TISABias()
    blk: nn.Module = GraphKRBlock()
    norm: nn.Module = nn.LayerNorm()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, S]
        f_ctx: jax.Array,  # [B, L_ctx, F]
        s_test: jax.Array,  # [B, L_test, S]
        valid_lens_ctx: Optional[jax.Array] = None,  # [B]
        valid_lens_test: Optional[jax.Array] = None,  # [B]
        training: bool = False,
        **kwargs,
    ):
        stack = lambda *args: jnp.concatenate(args, axis=-1)
        (B, N_t), N_c, K = s_test.shape[:-1], s_ctx.shape[1], self.k
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(N_c, B)
        mask = mask_from_valid_lens(N_c, valid_lens_ctx)
        s_send = jnp.where(mask, s_ctx, 1e6)  # masked values = far away for kNN
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        x_ctx, x_test = self.norm(self.embed_all(ctx)), self.norm(self.embed_all(test))
        knn = vmap(lambda r, s: self.k_nearest_senders(r, s, K))
        (s_cc, d_cc), (s_ct, d_ct) = knn(s_ctx, s_send), knn(s_test, s_send)
        s_cc = s_cc.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_c * K)
        s_ct = s_ct.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_t * K)
        nodes = jnp.vstack([x_ctx.reshape(B * N_c, -1), x_test.reshape(B * N_t, -1)])
        edges = stack(d_cc.flatten(), d_ct.flatten())
        g = GraphsTuple(
            nodes,
            edges,
            senders=stack(s_cc, s_ct),
            receivers=jnp.repeat(jnp.arange(B * (N_c + N_t)), K),
            n_node=jnp.array([B * (N_c + N_t)]),
            n_edge=jnp.array([B * (N_c + N_t) * K]),
            globals=None,
        )
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                bias = self.bias.copy()(edges[:, None, None]).squeeze()
                # NOTE: bucket_size is for numerical stability in
                # jax.ops.segment_* calls; this is typically only needed for
                # testing implementation correctness
                g = blk(g, training, bias=bias, bucket_size=kwargs.get("bucket_size"))
        x_t = g.nodes[-B * N_t :, :].reshape(B, N_t, -1)
        f_dist = self.head(x_t, training)
        f_mu, f_log_var = jnp.split(f_dist, 2, axis=-1)
        f_std = jnp.exp(f_log_var / 2)
        f_std = self.min_std + (1 - self.min_std) * f_std
        return f_mu, f_std
