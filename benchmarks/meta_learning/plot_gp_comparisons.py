#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from gp import build_2d_grid_dataloader
from jax import jit, random
from tqdm import tqdm

from dl4bi.core.train import load_ckpt
from dl4bi.meta_learning.data.utils import (
    inv_permute_L_in_BLD,
    unbatch_BLD,
)


def main(args):
    print(args.model_name_pattern)
    regex = re.compile(args.model_name_pattern)
    paths = [p for p in Path(args.dir).iterdir() if regex.search(p.name)]
    models = {}
    for path in paths:
        state, cfg = load_ckpt(path)
        model_cls_name = cfg.model._target_.split(".")[-1]
        models[model_cls_name] = {"state": state, "cfg": cfg}
    plot_shifted(models)


def plot_shifted(models, shift: float = 10.0, num_samples: int = 1):
    cfg = models["ScanTNPKR"]["cfg"]
    cfg.data.batch_size = 1
    for axis in cfg.data.s:
        axis.start += shift
        axis.stop += shift
    dataloader = build_2d_grid_dataloader(cfg.data, cfg.kernel)
    rng = random.key(cfg.seed)
    rng_data, rng = random.split(rng)
    batches = dataloader(rng_data)
    num_models = len(models)
    for i in tqdm(range(1, num_samples + 1)):
        rng_i, rng = random.split(rng)
        fig, axes = plt.subplots(3, len(models), figsize=(3 * num_models, 9))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        batch, _ = next(batches)
        for j, (model_cls_name, d) in enumerate(models.items()):
            state = d["state"]
            output = state.apply_fn(
                {"params": state.params, **state.kwargs},
                **batch,
                rngs={"extra": rng_i},
            )
            if isinstance(output, tuple):
                output, _ = output  # latent output not used here
            _plot(
                name=model_cls_name,
                batch=batch,
                output=output,
                f_pred_axis=axes[1, j],
                f_std_axis=axes[2, j],
                task_axis=None if j else axes[0, 1],
                ground_truth_axis=None if j else axes[0, 2],
            )
        axes[1, 0].set_ylabel("Prediction", fontsize=20)
        axes[2, 0].set_ylabel("Uncertainty", fontsize=20)
        for ax in axes.flatten():
            if not ax.has_data():
                ax.axis("off")
        fig.subplots_adjust(
            left=0.05,
            right=0.95,
            bottom=0.05,
            top=0.95,
            wspace=0.05,
            hspace=0.05,
        )
        plt.tight_layout()
        plt.savefig(f"sample_{i}.png")
        plt.clf()


def _plot(
    name,
    batch,
    output,
    f_pred_axis,
    f_std_axis,
    task_axis=None,
    ground_truth_axis=None,
):
    inv_p = batch.inv_permute_idx
    L = inv_p.shape[0] if inv_p.ndim == 1 else inv_p.shape[1]
    f_ctx = jnp.where(batch.mask_ctx[..., None], batch.f_ctx, jnp.nan)
    f_pred, f_std, f_test = output.mu, output.std, batch.f_test
    if f_std.shape[-1] > 1:  # e.g. uncertainty per RGB channel
        f_std = f_std.mean(axis=-1, keepdims=True)
    if batch.mask_test is not None:
        f_test = jnp.where(batch.mask_test[..., None], f_test, jnp.nan)
    arrays = unbatch_BLD([f_ctx, f_test, f_pred, f_std], L)
    arrays = inv_permute_L_in_BLD(arrays, batch.inv_permute_idx)
    reshape = jit(lambda v: v.reshape(*batch.s_shape[1:-1], v.shape[-1]).squeeze())
    f_ctx, f_test, f_pred, f_std = map(reshape, arrays)
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap.set_bad("grey")
    cmap_std = mpl.colormaps.get_cmap("plasma")
    kwargs = dict(cmap=cmap, interpolation="none")
    kwargs_std = dict(cmap=cmap_std, interpolation="none")
    f_pred_axis.imshow(f_pred, **kwargs)
    f_std_axis.imshow(f_std, **kwargs_std)
    f_std_axis.set_xlabel(name, fontsize=20)
    if task_axis:
        task_axis.set_title("Task", fontsize=20)
        task_axis.imshow(f_ctx, **kwargs)
    if ground_truth_axis:
        ground_truth_axis.set_title("Ground Truth", fontsize=20)
        ground_truth_axis.imshow(f_test, **kwargs)


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
        "-p",
        "--model_name_pattern",
        default=r".*\.ckpt",
        help="Load models that match this pattern.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main(parse_args(sys.argv))
