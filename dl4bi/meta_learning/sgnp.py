from functools import partial
from typing import Callable, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
from jraph import GraphsTuple

from ..core.gnn import EdgeBiasedGAT
from ..core.knn import STkNN, kNN
from ..core.mlp import MLP
from .model_output import DiagonalMVNOutput


class SGNP(nn.Module):
    """Sparse Graph Neural Process (SGNP).

    Args:
        knn: A kNN module to use for constructing the graphs.
        graph: An optional graph that, when provided, will be used for every
            example, i.e. the nodes will be replaced while the adjacency
            structure and edge weights remain the same. When the graph
            structure remains constant, this precludes the need to find
            kNN neighbors and build a graph for each example, which can
            dramatically speed up training.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        norm: A module used for normalization of input features before the GNN.
        gnn: A graph neural network used to update node embeddings.
        head: Transforms the test nodes into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.

    Returns:
        An instance of the `SGNP` model.
    """

    knn: Union[kNN, STkNN] = kNN()
    graph: Optional[GraphsTuple] = None
    embed_s: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    norm: nn.Module = nn.LayerNorm()
    gnn: nn.Module = EdgeBiasedGAT()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_conditional_np

    @nn.compact
    def __call__(
        self,
        s_ctx: jax.Array,  # [B, L_ctx, S]
        f_ctx: jax.Array,  # [B, L_ctx, F]
        s_test: jax.Array,  # [B, L_test, S]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        training: bool = False,
        **kwargs,
    ):
        stack = lambda *args: jnp.concatenate([x for x in args if x.size > 0], axis=-1)
        (B, N_t), N_c = s_test.shape[:-1], s_ctx.shape[1]
        f_test = jnp.zeros([*s_test.shape[:-1], f_ctx.shape[-1]])
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = stack(self.embed_obs(obs), self.embed_s(s_ctx), self.embed_f(f_ctx))
        test = stack(self.embed_obs(unobs), self.embed_s(s_test), self.embed_f(f_test))
        n_ctx, n_test = self.norm(self.embed_all(ctx)), self.norm(self.embed_all(test))
        # nodes for the graph are all the context nodes followed by all test nodes
        nodes = jnp.vstack([n_ctx.reshape(B * N_c, -1), n_test.reshape(B * N_t, -1)])
        g = self.graph  # if a graph is provided, reuse it, updating only nodes
        if g is None:
            g = build_graph(self.knn, s_ctx, s_test, mask_ctx)
        g = g._replace(nodes=nodes)
        g = self.gnn(g, training, **kwargs)
        n_t = g.nodes[-B * N_t :, :].reshape(B, N_t, -1)
        output = self.head(n_t, training)
        return self.output_fn(output)


@partial(jit, static_argnames=("knn",))
def build_graph(
    knn: Union[kNN, STkNN],
    s_ctx: jax.Array,  # [B, L_ctx, S]
    s_test: jax.Array,  # [B, L_test, S]
    mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
):
    """Builds a single graph from a batch of tasks.

    This constructor assumes that when nodes are added, they will be a
    single array of all context points followed by all test points, i.e.
    `nodes` will be of shape `[B * N_c + B * N_t, D]`.
    """
    (B, N_t), N_c, K = s_test.shape[:-1], s_ctx.shape[1], knn.k
    s_send = s_ctx
    if mask_ctx is not None:  # masked vlaues = far away for kNN
        s_send = jnp.where(mask_ctx[..., None], s_ctx, jnp.inf)
    s_cc, d_cc = knn(s_ctx, s_send)  # s_cc: [B, Q_c, K], d_cc: [B, Q_c, K, D]
    s_ct, d_ct = knn(s_test, s_send)  # s_ct: [B, Q_t, K], d_ct: [B, Q_t, K, D]
    D = d_cc.shape[-1]
    edges = jnp.vstack([d_cc.reshape(-1, D), d_ct.reshape(-1, D)])
    edge_mask = jnp.all(jnp.isfinite(edges), axis=-1)
    # convert indices from batch level to graph level
    s_cc = s_cc.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_c * K)
    s_ct = s_ct.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_t * K)
    return GraphsTuple(
        nodes=None,
        edges=edges,
        receivers=jnp.repeat(jnp.arange(B * (N_c + N_t)), K),
        senders=jnp.hstack([s_cc, s_ct]),
        n_node=jnp.array([B * (N_c + N_t)]),
        n_edge=jnp.array([B * (N_c + N_t) * K]),
        globals={"edge_mask": edge_mask},
    )
