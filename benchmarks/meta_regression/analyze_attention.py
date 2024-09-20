#!/usr/bin/env python3
"""
This file analyzes attention stored Flax module intermediates variables.

NOTE: This requires that the following tensors have been stored from the
relevant Flax module:
```
self.sow("intermediates", "cross_attn", cross_attn)
self.sow("intermediates", "self_attn", cross_attn)
self.sow("intermediates", "qvs", qvs)
self.sow("intermediates", "kvs", kvs)
```
"""

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.gridl4biec as gridl4biec
import matplotlib.pyplot as plt
from jax import random
from pygifsicle import optimize
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior

import dl4bi.meta_regression.train_utils as tu


def main(args):
    rng = random.key(args.seed)
    rng_gp, rng_extra = random.split(rng)
    s_min, s_max, num_ctx, num_test, ls = -2, 2, 10, 128, 0.2
    s_ctx = jnp.linspace(s_min + 0.1, s_max - 0.1, num_ctx)[:, None]  # [L_ctx, 1]
    s_test = jnp.linspace(s_min, s_max, num_test)[:, None]  # [L_test, 1]
    gp = GP(rbf, ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng_gp, jnp.vstack([s_ctx, s_test]), batch_size=1)
    f_ctx, f_test = f[0, :num_ctx, :], f[0, num_ctx:, :]
    state, _ = tu.load_ckpt(Path(args.ckpt_path))
    (f_mu, f_std, *_), vars = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_ctx[None, ...],  # add batch dim
        f_ctx[None, ...],
        s_test[None, ...],
        mutable="intermediates",
        rngs={"extra": rng_extra},
    )
    path_leaf_tpls = jtu.tree_leaves_with_path(vars["intermediates"])
    ## Plot attns
    cross_attns = [x for path, x in path_leaf_tpls if "cross_attn" in jtu.keystr(path)]
    self_attns = [x for path, x in path_leaf_tpls if "self_attn" in jtu.keystr(path)]
    cross_attns = jnp.stack(cross_attns).squeeze()  # [N, H, L_test, L_ctx]
    self_attns = jnp.stack(self_attns).squeeze()  # [N, H, L_ctx, L_ctx]
    fig = plot_attn(cross_attns, "Cross-Attention")
    save(fig, "cross_attn.png")
    fig = plot_attn(self_attns, "Self-Attention")
    save(fig, "self_attn.png")

    ## Plot qvs/kvs
    qvs = [x for path, x in path_leaf_tpls if "qvs" in jtu.keystr(path)]
    kvs = [x for path, x in path_leaf_tpls if "kvs" in jtu.keystr(path)]
    qvs = jnp.stack(qvs).squeeze()  # [N, L_test, D]
    kvs = jnp.stack(kvs).squeeze()  # [N, L_ctx, D]
    fig = plot_embed(qvs, "Query Embeddings (Test Points)")
    save(fig, "qvs.png")
    fig = plot_embed(kvs, "Key Embeddings (Context Points)")
    save(fig, "kvs.png")

    ## Plot sample
    fig = tu.plot_posterior_predictive(
        s_ctx.squeeze(),  # squeeze out first and last dims in [B=1, L, D_S=1]
        f_ctx.squeeze(),
        s_test.squeeze(),
        f_test.squeeze(),
        f_mu.squeeze(),
        f_std.squeeze(),
    )
    query_id = 37
    s, f = s_test[query_id].squeeze(), f_mu[0, query_id].squeeze()
    weights = cross_attns[-1, -1, query_id, ...]  # last layer, last head
    fig = add_cross_attn_lines(fig, s, f, s_ctx.squeeze(), f_ctx.squeeze(), weights)
    title = f"RBF GP sample with (var: 1.0, ls: {ls}), seed {args.seed}"
    fig.suptitle(title)
    save(fig, "sample.png")

    ## Save attention gif
    attn = cross_attns[-1, -1, ...]  # last layer, last head
    save_attn_gif(
        attn,
        s_ctx.squeeze(),
        f_ctx.squeeze(),
        s_test.squeeze(),
        f_test.squeeze(),
        f_mu.squeeze(),
        f_std.squeeze(),
        title,
    )


def save(fig, name: str):
    fig.savefig(name, dpi=150)
    plt.clf()
    plt.close(fig)


def plot_attn(attn: jax.Array, title: str):
    N, H, L_test, L_ctx = attn.shape
    attn -= attn.min()  # normalize values to [0, 1] across layers
    attn /= attn.max()
    fig = plt.figure(figsize=(H * 2 + 1, N * 2))
    gs = gridl4biec.GridSpec(N, H + 1, width_ratios=[*[1] * H, 0.1])
    for n in range(N):
        for h in range(H):
            ax = fig.add_subplot(gs[n, h])
            if h == 0:
                ax.set_ylabel(f"Layer {n+1}")
            if n == N - 1:
                ax.set_xlabel(f"Head {h+1}")
            im = ax.imshow(attn[n, h], aspect="auto", cmap="inferno")
    cax = fig.add_subplot(gs[:, H])
    fig.colorbar(im, cax=cax)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_embed(x: jax.Array, title: str):
    N, L, D = x.shape
    x -= x.min()  # normalize values to [0, 1] across layers
    x /= x.max()
    fig = plt.figure(figsize=(6, 12))
    gs = gridl4biec.GridSpec(N, 2, width_ratios=[1, 0.05])
    for n in range(N):
        ax = fig.add_subplot(gs[n, 0])
        ax.set_ylabel(f"Layer {n}")
        if n == N - 1:
            ax.set_xlabel("Embedding")
        im = ax.imshow(x[n].T, aspect="auto", cmap="viridis")
    cax = fig.add_subplot(gs[:, 1])
    fig.colorbar(im, cax=cax)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def save_attn_gif(
    attn: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    f_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    title: str,
):
    paths = []
    for query_id in range(s_test.shape[0]):
        fig = tu.plot_posterior_predictive(s_ctx, f_ctx, s_test, f_test, f_mu, f_std)
        s, f, weights = s_test[query_id], f_mu[query_id], attn[query_id]
        fig = add_cross_attn_lines(fig, s, f, s_ctx, f_ctx, weights)
        fig.suptitle(title)
        paths += [f"/tmp/{query_id}.png"]
        save(fig, paths[-1])
    frames = jnp.stack([iio.imread(p) for p in paths], axis=0)
    iio.imwrite("attn.gif", frames)
    optimize("attn.gif")


def add_cross_attn_lines(
    fig,
    s: jax.Array,
    f: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    weights: jax.Array,
):
    ax = fig.axes[0]
    ax.plot(s, f, "ro")
    for sc, fc, wc in zip(s_ctx, f_ctx, weights):
        ax.plot([s, sc], [f, fc], color="r", alpha=0.75, linewidth=wc * 5)
    return fig


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ckpt_path",
        help="Path to a 1D GP RBF model checkpoint.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=7,
        help="Root seed for all random operations.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
