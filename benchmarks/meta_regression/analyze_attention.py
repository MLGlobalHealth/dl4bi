#!/usr/bin/env python3
"""
This file analyzes attention stored Flax module intermediates variables.
"""

import argparse
import sys

import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def main(args):
    d = jnp.load(args.intermediates_path, allow_pickle=True).item()
    # batch, preds = jnp.load(args.predictions_path, allow_pickle=True).item()
    batch = jnp.load(args.predictions_path, allow_pickle=True).item()
    # s_ctx, _, valid_lens_ctx, *_ = batch
    s_ctx, valid_lens_ctx = batch["s_ctx"], batch["valid_lens_ctx"]
    attns = [
        attn
        for path, attn in jtu.tree_leaves_with_path(d)
        if args.attn_key in jtu.keystr(path)
    ]
    v = batch["valid_lens_ctx"][0]
    # v = valid_lens_ctx[0]
    s_ctx = s_ctx[0, :v, 0]  # [B=1, L_ctx, D_S=1]
    idx = jnp.argsort(s_ctx)
    s_ctx = s_ctx[idx]  # sort s_ctx
    num_layers, num_heads = len(attns), attns[0].shape[1]  # [B, H, L_ctx, L_ctx]
    fig = plt.figure(figsize=(num_heads * 2, num_layers * 2))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(num_layers, num_heads),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )
    for i in range(num_layers):
        for h in range(num_heads):
            k = i * num_heads + h
            attn_head = attns[i][0, h, :, :v]
            # valid_lens apply to queries and keys in self_attn
            if args.attn_key == "self_attn":
                attn_head = attn_head[:v, :]
            attn_head = attn_head[:, idx][idx, :]
            im = grid[k].imshow(attn_head, cmap="inferno")
            grid[k].set_ylabel(f"Layer {i+1}")
            grid[k].set_xlabel(f"Head {h+1}")
    grid[0].cax.colorbar(im)
    plt.tight_layout()
    plt.suptitle(args.title)
    plt.savefig("test_attn.pdf", dpi=150)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "intermediates_path",
        default="intermediates.npy",
        help="Path to intermediates numpy file.",
    )
    parser.add_argument(
        "predictions_path",
        default="predictions.npy",
        help="Path to numpy file with (batch, preds).",
    )
    parser.add_argument(
        "-k",
        "--attn_key",
        default="self_attn",
        help="Intermediate attention key name.",
    )
    parser.add_argument(
        "-t",
        "--title",
        default="Attention",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args)
