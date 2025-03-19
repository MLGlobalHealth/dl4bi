from functools import partial
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit
from jraph import GraphsTuple

from ..core.gnn import GraphAttentionBlock
from ..core.mlp import MLP
from ..core.model_output import DiagonalMVNOutput
from ..core.sim import delta_time, l2_dist
from ..core.utils import exists, safe_stack
from .steps import likelihood_train_step, likelihood_valid_step
from .utils import first_shape


class SGNP(nn.Module):
    r"""Sparse Graph Neural Process (SGNP).

    Args:
        k: Number of neighbors to use per point.
        num_blks: Number of `KRBlocks` to use.
        num_reps: Number of times to repeat each `KRBlock`.
        embed_s: A module that embeds the index set prior to embedding with
            function values.
        embed_f: A module that embeds function values prior to embedding with
            the index set.
        embed_obs: A module that creates embeddings for observed (context) and
            unobserved (test) points.
        embed_all: A module that jointly embeds `obs`, `s`, and `f` embeddings.
        x_sim: Similarity function for fixed effects. None-finite values are masked.
        s_sim: Similarity function for spatial effects. Non-finite values are masked.
        t_sim: Similarity function for temporal effects. Non-finite values are masked.
        scale_x_sim: A scalar to use for `x` similarity values when combining to get
            k-nearest neighbors.
        scale_s_sim: A scalar to use for `s` similarity values when combining to get
            k-nearest neighbors.
        scale_t_sim: A scalar to use for `t` similarity values when combining to get
            k-nearest neighbors.
        causal_t: Whether to enforce temporal causality in attention calculations.
        x_bias: Bias module for fixed similarity values.
        s_bias: Bias module for spatial similarity values.
        t_bias: Bias module for temporal similarity values.
        norm: A module used for normalization between blocks.
        blk: A block to use for each layer and each repetition.
        head: Transforms the tokens into model output.
        output_fn: A function that transforms the model output into
            a form that can be consumed by loss functions.
        graph: An optional graph to reuse, updating only nodes.
        train_step: What training step to use.
        valid_step: What validation step to use.

    Returns:
        An instance of the `SGNP` model.

    .. note::
        If `x_ctx` and `x_test` are provided but `x_sim` is `None`, `x_ctx` and `x_test`
        will be incorporated into embeddings, but will not be used for selecting the k
        nearest neighbors. The same follows for the corresponding `s` and `t` values.

    .. note::
        When selecting the k nearest neighbors, the distance formula used is:
        $d^2 = (\text{scale}_x\cdot d_x)^2+(\text{scale}_s\cdot d_s)^2+(\text{scale}_t\cdot d_t)^2$
    """

    k: int = 16
    num_blks: int = 6
    num_reps: int = 1
    embed_x: Callable = lambda x: x
    embed_s: Callable = lambda x: x
    embed_t: Callable = lambda x: x
    embed_f: Callable = lambda x: x
    embed_obs: nn.Module = nn.Embed(2, 4)
    embed_all: nn.Module = MLP([256, 128, 64], nn.gelu)
    x_sim: Optional[Callable] = l2_dist
    s_sim: Optional[Callable] = l2_dist
    t_sim: Optional[Callable] = delta_time
    scale_x_sim: float = 1.0
    scale_s_sim: float = 1.0
    scale_t_sim: float = 1.0
    causal_t: bool = True
    x_bias: Optional[Callable] = None
    s_bias: Optional[Callable] = None
    t_bias: Optional[Callable] = None
    norm: nn.Module = nn.LayerNorm()
    blk: nn.Module = GraphAttentionBlock()
    head: nn.Module = MLP([256, 64, 2], nn.gelu)
    output_fn: Callable = DiagonalMVNOutput.from_activations
    graph: Optional[GraphsTuple] = None
    train_step: Callable = likelihood_train_step
    valid_step: Callable = likelihood_valid_step

    @nn.compact
    def __call__(
        self,
        x_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_x]
        s_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
        t_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_t]
        f_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_f]
        mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
        x_test: Optional[jax.Array] = None,  # [B, L_test, D_x]
        s_test: Optional[jax.Array] = None,  # [B, L_test, D_s]
        t_test: Optional[jax.Array] = None,  # [B, L_test, D_t]
        training: bool = False,
        **kwargs,
    ):
        norm = nn.LayerNorm()
        test_shape = first_shape([x_test, s_test, t_test])
        f_test = jnp.zeros((*test_shape[:-1], f_ctx.shape[-1]))
        obs = jnp.ones(f_ctx.shape[:-1], dtype=jnp.uint8)
        unobs = jnp.zeros(f_test.shape[:-1], dtype=jnp.uint8)
        ctx = safe_stack(
            self.embed_obs(obs),
            self.embed_x(x_ctx),
            self.embed_s(s_ctx),
            self.embed_t(t_ctx),
            self.embed_f(f_ctx),
        )
        test = safe_stack(
            self.embed_obs(unobs),
            self.embed_x(x_test),
            self.embed_s(s_test),
            self.embed_t(t_test),
            self.embed_f(f_test),
        )
        qvs, kvs = map(lambda x: norm(self.embed_all(x)), (test, ctx))
        # nodes for the graph are all the context nodes followed by all test nodes
        (B, N_t), N_c = test_shape[:-1], f_ctx.shape[1]
        nodes = jnp.vstack([kvs.reshape(B * N_c, -1), qvs.reshape(B * N_t, -1)])
        g = self.graph  # if a graph is provided, reuse it, updating only nodes
        if g is None:
            g = build_graph(
                x_ctx,
                s_ctx,
                t_ctx,
                mask_ctx,
                x_test,
                s_test,
                t_test,
                self.k,
                self.x_sim,
                self.s_sim,
                self.t_sim,
                self.causal_t,
                self.scale_x_sim,
                self.scale_s_sim,
                self.scale_t_sim,
            )
        g = g._replace(nodes=nodes)
        edge_mask = g.globals.get("edge_mask")
        for _ in range(self.num_blks):
            blk = self.blk.copy()
            for _ in range(self.num_reps):
                bias = 0
                if self.x_bias is not None:
                    bias += self.x_bias.copy()(g.edges["d_x"], edge_mask)
                if self.s_bias is not None:
                    bias += self.s_bias.copy()(g.edges["d_s"], edge_mask)
                if self.t_bias is not None:
                    bias += self.t_bias.copy()(g.edges["d_t"], edge_mask)
                # NOTE: bucket_size is for numerical stability in
                # jax.ops.segment_* calls; this is typically only needed for
                # testing implementation correctness
                g = blk(g, training, bias=bias, bucket_size=kwargs.get("bucket_size"))
        nodes_test = g.nodes[-B * N_t :, :].reshape(B, N_t, -1)
        output = self.head(nodes_test, training)
        return self.output_fn(output)


@partial(
    jit,
    static_argnames=(
        "k",
        "x_sim",
        "s_sim",
        "t_sim",
        "causal_t",
        "scale_x_sim",
        "scale_s_sim",
        "scale_t_sim",
    ),
)
def build_graph(
    x_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_x]
    s_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_s]
    t_ctx: Optional[jax.Array] = None,  # [B, L_ctx, D_t]
    mask_ctx: Optional[jax.Array] = None,  # [B, L_ctx]
    x_test: Optional[jax.Array] = None,  # [B, L_test, D_x]
    s_test: Optional[jax.Array] = None,  # [B, L_test, D_s]
    t_test: Optional[jax.Array] = None,  # [B, L_test, D_t]
    k: int = 16,
    x_sim: Optional[Callable] = None,
    s_sim: Optional[Callable] = None,
    t_sim: Optional[Callable] = None,
    causal_t: bool = True,
    scale_x_sim: float = 1.0,
    scale_s_sim: float = 1.0,
    scale_t_sim: float = 1.0,
):
    ctx, test = [x_ctx, s_ctx, t_ctx], [x_test, s_test, t_test]
    r_ctx = map(lambda v: _mask(v, mask_ctx), ctx)
    args = [k, x_sim, s_sim, t_sim, causal_t, scale_x_sim, scale_s_sim, scale_t_sim]
    idx_cc, d_x_cc, d_s_cc, d_t_cc = approx_knn(*ctx, *r_ctx, *args)
    idx_ct, d_x_ct, d_s_ct, d_t_ct = approx_knn(*test, *r_ctx, *args)
    # convert indices from batch level to graph level
    ctx_shape, test_shape = first_shape(ctx), first_shape(test)
    (B, N_t), N_c = test_shape[:-1], ctx_shape[1]
    idx_cc = idx_cc.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_c * k)
    idx_ct = idx_ct.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_t * k)
    edges = {}
    x_mask = s_mask = t_mask = jnp.array([True])
    if exists(d_x_cc, d_x_ct):
        edges["d_x"] = _batch_edges(d_x_cc, d_x_ct)
        x_mask = jnp.isfinite(edges["d_x"])
    if exists(d_s_cc, d_s_ct):
        edges["d_s"] = _batch_edges(d_s_cc, d_s_ct)
        s_mask = jnp.isfinite(edges["d_s"])
    if exists(d_t_cc, d_t_ct):
        edges["d_t"] = _batch_edges(d_t_cc, d_t_ct)
        t_mask = jnp.isfinite(edges["d_t"])
    edge_mask = x_mask & s_mask & t_mask
    return GraphsTuple(
        nodes=None,
        edges=edges,
        receivers=jnp.repeat(jnp.arange(B * (N_c + N_t)), k),
        senders=jnp.hstack([idx_cc, idx_ct]),
        n_node=jnp.array([B * (N_c + N_t)]),
        n_edge=jnp.array([B * (N_c + N_t) * k]),
        globals={"edge_mask": edge_mask},
    )


def _mask(a: Optional[jax.Array], mask: Optional[jax.Array]):
    if a is None:
        return None
    if mask is None:
        return a
    return jnp.where(mask[..., None], a, jnp.inf)


def _batch_edges(d_cc: jax.Array, d_ct: jax.Array):
    D = d_cc.shape[-1]
    return jnp.vstack([d_cc.reshape(-1, D), d_ct.reshape(-1, D)])


@partial(
    jit,
    static_argnames=(
        "k",
        "x_sim",
        "s_sim",
        "t_sim",
        "causal_t",
        "scale_x_sim",
        "scale_s_sim",
        "scale_t_sim",
        "num_q_parallel",
        "recall_target",
    ),
)
def approx_knn(
    q_x: Optional[jax.Array],
    q_s: Optional[jax.Array],
    q_t: Optional[jax.Array],
    r_x: Optional[jax.Array],
    r_s: Optional[jax.Array],
    r_t: Optional[jax.Array],
    k: int,
    x_sim: Optional[Callable],
    s_sim: Optional[Callable],
    t_sim: Optional[Callable],
    causal_t: bool = False,
    scale_x_sim: float = 1.0,
    scale_s_sim: float = 1.0,
    scale_t_sim: float = 1.0,
    num_q_parallel: int = 1024,
    recall_target: float = 0.95,
):
    def process_batch(q: dict):
        d_x = d_s = d_t = 0
        if "q_x" in q and exists(x_sim):
            d_x = x_sim(q["q_x"][None, ...], r_x).squeeze()  # [R]
        if "q_s" in q and exists(s_sim):
            d_s = s_sim(q["q_s"][None, ...], r_s).squeeze()  # [R]
        if "q_t" in q and exists(t_sim):
            d_t = t_sim(q["q_t"][None, ...], r_t).squeeze()  # [R]
            if causal_t:
                d_t = jnp.where(d_t <= 0, d_t, jnp.inf)
        d_sq = (
            (scale_x_sim * d_x) ** 2
            + (scale_s_sim * d_s) ** 2
            + (scale_t_sim * d_t) ** 2
        )
        _, idx = jax.lax.approx_min_k(d_sq, k, recall_target=recall_target)
        d_x, d_s, d_t = map(lambda v: _idx_or_none(v, idx), [d_x, d_s, d_t])
        return idx, d_x, d_s, d_t

    q = {f"q_{k}": v for k, v in [("x", q_x), ("s", q_s), ("t", q_t)] if v is not None}
    idx, d_x, d_s, d_t = jax.lax.map(process_batch, q, batch_size=num_q_parallel)
    return idx, d_x, d_s, d_t


def _idx_or_none(a, idx: jax.Array):
    if isinstance(a, jax.Array):
        return a[idx]
    return None
