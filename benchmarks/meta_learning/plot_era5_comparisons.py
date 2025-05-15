#!/usr/bin/env python3
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from era5 import build_dataloaders, t_to_label
from jax import jit, random
from matplotlib.colors import Normalize
from tqdm import tqdm

from dl4bi.core.train import load_ckpt
from dl4bi.core.utils import nan_pad
from dl4bi.meta_learning.data.spatiotemporal import _inv_permute_Ls
from dl4bi.meta_learning.data.utils import unbatch_BLD


def main(args):
    models = {}
    for path in Path(args.dir).glob("*.ckpt"):
        state, cfg = load_ckpt(path)
        model_cls_name = cfg.model._target_.split(".")[-1]
        models[model_cls_name] = {"state": state, "cfg": cfg}
    plot(models, args.num_ctx_per_t, args.num_samples)


def plot(
    models,
    num_ctx_per_t: int = 32,
    num_samples: int = 16,
):
    key = list(models.keys())[0]
    cfg = models[key]["cfg"]  # cfg.data should be the same for all
    cfg.data.batch_size = 1
    cfg.data.num_ctx_min_per_t = num_ctx_per_t
    cfg.data.num_ctx_max_per_t = num_ctx_per_t
    *_, dataloader = build_dataloaders(**cfg.data)
    rng = random.key(cfg.seed)
    rng_data, rng = random.split(rng)
    batches = dataloader(rng_data)
    Path("samples").mkdir(exist_ok=True)
    for i in tqdm(range(1, num_samples + 1)):
        rng_i, rng = random.split(rng)
        batch, revert = next(batches)
        f_min = min(batch.f_ctx.min(), batch.f_test.min())
        f_max = max(batch.f_ctx.max(), batch.f_test.max())
        f_std_min, f_std_max = jnp.inf, -jnp.inf
        for j, (model_cls_name, d) in enumerate(models.items()):
            state = d["state"]
            output = state.apply_fn(
                {"params": state.params, **state.kwargs},
                **batch,
                rngs={"extra": rng_i},
            )
            if isinstance(output, tuple):
                output, _ = output  # latent output not used here
            f_min, f_max = min(f_min, output.mu.min()), max(f_max, output.mu.max())
            f_std_min = min(f_std_min, output.std.min())
            f_std_max = max(f_std_max, output.std.max())
            models[model_cls_name]["output"] = output
        batch = replace(
            batch,
            s_ctx=revert["s"](batch.s_ctx),
            s_test=revert["s"](batch.s_test),
            t_ctx=t_to_label(revert["t"](batch.t_ctx)),
            t_test=t_to_label(revert["t"](batch.t_test)),
        )
        (T_b, L), (B, _, D_f) = batch.inv_permute_idx.shape, batch.f_ctx.shape
        # fill in masked values with nan
        f_ctx = jnp.where(batch.mask_ctx[..., None], batch.f_ctx, jnp.nan)
        # reintroduce timestep and nan pad each time step to full size
        f_ctx = f_ctx.reshape(B, T_b - 1, -1, D_f)
        f_ctx = nan_pad(f_ctx, axis=2, L=L)
        f_test = batch.f_test[:, None]  # add time step dim
        f_ctx_orig = f_ctx
        f_ctx, f_test = _inv_permute_Ls(f_ctx, f_test, batch.inv_permute_idx)
        reshape_s = jit(lambda v: v.reshape(*v.shape[:2], *batch.s_dims, v.shape[-1]))
        f_ctx, f_test = map(reshape_s, [f_ctx, f_test])
        name = {
            "ScanTNPKR": "BSA-TNP",
            "BSATNP": "BSA-TNP",
            "TNPD": "TNP-D",
            "TETNP": "PT-TE-TNP",
        }
        cmap = mpl.colormaps.get_cmap("Spectral_r")
        cmap.set_bad("grey")
        norm = Normalize(f_min, f_max)
        norm_std = Normalize(f_std_min, f_std_max)
        kwargs = dict(cmap=cmap, norm=norm, interpolation="none")
        std_kwargs = dict(cmap="plasma", norm=norm_std, interpolation="none")
        fig, axs = plt.subplots(3, T_b, figsize=(5 * T_b, 3 * 5), squeeze=False)
        t_ctx = batch.t_ctx.reshape(B, T_b - 1, -1, 1)
        t_test = batch.t_test[0, 0, 0].item()
        for j in range(T_b - 1):
            axs[0, j].imshow(f_ctx[i, j], **kwargs)
            axs[0, j].set_title(f"t={t_ctx[0, j, 0, 0].item()}", fontsize=30)
            if j == 0:
                axs[0, j].set_xlabel("Context", fontsize=30)
        axs[0, -1].imshow(f_test[0, 0], **kwargs)
        axs[0, -1].set_title(f"t={t_test}", fontsize=30)
        axs[0, -1].set_xlabel("Ground Truth", fontsize=30)
        for j, model_cls_name in enumerate(["TNPD", "TETNP", "ScanTNPKR"]):
            # for j, (model_cls_name, d) in enumerate(models.items()):
            # output = d["output"]
            output = models[model_cls_name]["output"]
            f_pred, f_std = output.mu[:, None], output.std[:, None]
            # invert permutation of the flattened spatial dim, L, by time step
            _, f_pred = _inv_permute_Ls(f_ctx_orig, f_pred, batch.inv_permute_idx)
            _, f_std = _inv_permute_Ls(f_ctx_orig, f_std, batch.inv_permute_idx)
            # reshape to original spatial dims
            f_pred, f_std = map(reshape_s, [f_pred, f_std])
            axs[1, j + 1].imshow(f_pred[0, 0], **kwargs)
            # axs[1, j + 1].set_title(f"t={t_test}", fontsize=30)
            axs[2, j + 1].set_xlabel(name[model_cls_name], fontsize=30)
            axs[2, j + 1].imshow(f_std[0, 0], **std_kwargs)
            if j == 0:
                axs[1, j + 1].set_ylabel("Prediction", fontsize=30)
                axs[2, j + 1].set_ylabel("Uncertainty", fontsize=30)
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            if not ax.has_data():
                ax.axis("off")
        fig.subplots_adjust(
            left=0.0,  # rightmost subplot left edge
            right=1.0,  # leftmost subplot right edge
            bottom=0.04,  # top of bottom row
            top=0.96,  # bottom of top row
            wspace=0.01,  # width space between columns
            hspace=0.05,  # height space between rows
        )
        # plt.tight_layout()
        plt.savefig(f"samples/era5_{i}.png")
        plt.clf()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dir",
        help="Directory with model checkpoints to compare.",
    )
    parser.add_argument(
        "--num_ctx_per_t",
        default=128,
        type=int,
        help="Number of context points.",
    )
    parser.add_argument(
        "--num_samples",
        default=16,
        type=int,
        help="Number of samples to plot.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main(parse_args(sys.argv))
