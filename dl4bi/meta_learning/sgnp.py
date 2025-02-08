from collections.abc import Callable
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jraph import GraphsTuple

from ..core import MLP, EdgeBiasedGAT, kNN, mask_from_valid_lens
from .transform import diagonal_mvn


class SGNP(nn.Module):
    """SGNP."""

    knn: Callable = kNN()
    num_blks: int = 6
    num_reps: int = 1
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    norm: nn.Module = nn.LayerNorm()
    gnn: nn.Module = EdgeBiasedGAT()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = diagonal_mvn

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
        stack = lambda *args: jnp.concatenate([x for x in args if x.size > 0], axis=-1)
        (B, N_t), N_c, K = s_test.shape[:-1], s_ctx.shape[1], self.knn.k
        if valid_lens_ctx is None:
            valid_lens_ctx = jnp.repeat(N_c, B)
        mask = mask_from_valid_lens(N_c, valid_lens_ctx)
        s_send = jnp.where(mask, s_ctx, jnp.inf)  # masked values = far away for kNN
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        x_ctx, x_test = self.norm(self.embed_all(ctx)), self.norm(self.embed_all(test))
        # nodes for the graph are all the context nodes followed by all test nodes
        nodes = jnp.vstack([x_ctx.reshape(B * N_c, -1), x_test.reshape(B * N_t, -1)])
        g = kwargs.get("graph")
        if g is None:
            g = self.build_graph(nodes, s_ctx, s_test, s_send)
        else:  # reuse graph, replacing nodes
            g = g._replace(nodes=nodes)
        g = self.gnn(g, training, **kwargs)
        x_t = g.nodes[-B * N_t :, :].reshape(B, N_t, -1)
        f_dist = self.head(x_t, training)
        return self.output_fn(f_dist)

    def build_graph(
        self,
        nodes: jax.Array,
        s_ctx: jax.Array,
        s_test: jax.Array,
        s_send: jax.Array,
    ):
        """Builds a single graph from a batch of tasks.

        This assumes nodes represent all test points stacked after all context
        points, i.e. [B*N_c, B*N_t].
        """
        (B, N_t), N_c, K = s_test.shape[:-1], s_ctx.shape[1], self.knn.k
        s_cc, d_cc = self.knn(s_ctx, s_send)  # s_cc: [B, Q_c, K], d_cc: [B, Q_c, K, D]
        s_ct, d_ct = self.knn(s_test, s_send)  # s_ct: [B, Q_t, K], d_ct: [B, Q_t, K, D]
        D = d_cc.shape[-1]
        edges = jnp.vstack([d_cc.reshape(-1, D), d_ct.reshape(-1, D)])
        edge_mask = jnp.all(jnp.isfinite(edges), axis=-1)
        # convert indices from batch level to graph level
        s_cc = s_cc.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_c * K)
        s_ct = s_ct.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_t * K)
        g = GraphsTuple(
            nodes,
            edges,
            receivers=jnp.repeat(jnp.arange(B * (N_c + N_t)), K),
            senders=jnp.hstack([s_cc, s_ct]),
            n_node=jnp.array([B * (N_c + N_t)]),
            n_edge=jnp.array([B * (N_c + N_t) * K]),
            globals={"edge_mask": edge_mask},
        )
        return g
