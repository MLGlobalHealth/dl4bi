#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from gp import build_2d_grid_dataloader
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig
from tqdm import tqdm

from dl4bi.core.train import load_ckpt
from dl4bi.meta_learning.data.utils import (
    inv_permute_L_in_BLD,
    unbatch_BLD,
)


def main(args):
    models = {}
    for path in Path(args.dir).glob("*.ckpt"):
        state, cfg = load_ckpt(path)
        model_cls_name = cfg.model._target_.split(".")[-1]
        models[model_cls_name] = {"state": state, "cfg": cfg}
    plot(
        models,
        args.scale,
        args.shift,
        args.num_ctx,
        args.num_samples,
        args.num_points_per_unit,
    )


def plot(
    models,
    scale: float = 4.0,
    shift: float = 0.0,
    num_ctx: int = 32,
    num_samples: int = 16,
    num_points_per_unit: int = 16,
):
    key = list(models.keys())[0]
    cfg = models[key]["cfg"]  # cfg.data should be the same for all
    cfg.data.batch_size = 1
    half = scale / 2
    for axis in cfg.data.s:  # assumes coordinate axis is centered on origin
        axis.start = -half
        axis.stop = half
        axis.num = int(scale * num_points_per_unit)
        axis.start += shift
        axis.stop += shift
    cfg.data.num_ctx.min = num_ctx
    cfg.data.num_ctx.max = num_ctx
    models["ConvCNP"] = update_convcnp_grid(models["ConvCNP"], cfg.data.s)
    dataloader = build_2d_grid_dataloader(cfg.data, cfg.kernel)
    rng = random.key(cfg.seed)
    rng_data, rng = random.split(rng)
    batches = dataloader(rng_data)
    num_models = len(models)
    Path("samples").mkdir(exist_ok=True)
    for i in tqdm(range(1, num_samples + 1)):
        rng_i, rng = random.split(rng)
        batch, extra = next(batches)
        f_std_min, f_std_max = jnp.inf, -jnp.inf
        f_min, f_max = batch.f_test.min(), batch.f_test.max()
        for j, (model_cls_name, d) in enumerate(models.items()):
            state = d["state"]
            output = state.apply_fn(
                {"params": state.params, **state.kwargs},
                **batch,
                rngs={"extra": rng_i},
            )
            if isinstance(output, tuple):
                output, _ = output  # latent output not used here
            models[model_cls_name]["output"] = output
            f_std_min = min(f_std_min, output.std.min())
            f_std_max = max(f_std_max, output.std.max())
            f_max = max(f_max, output.mu.max())
            f_min = min(f_max, output.mu.min())
        f_norm = mpl.colors.Normalize(vmin=f_min, vmax=f_max)
        f_std_norm = mpl.colors.Normalize(vmin=f_std_min, vmax=f_std_max)
        fig, axes = plt.subplots(3, len(models), figsize=(3 * num_models, 9))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        for j, (model_cls_name, d) in enumerate(models.items()):
            name = {
                "ConvCNP": "ConvCNP",
                "BSATNP": "BSA-TNP",
                "TNPD": "TNP-D",
                "TETNP": "PT-TE-TNP",
            }
            _plot(
                name=name[model_cls_name],
                batch=batch,
                output=d["output"],
                f_pred_axis=axes[1, j],
                f_std_axis=axes[2, j],
                f_norm=f_norm,
                f_std_norm=f_std_norm,
                task_axis=None if j else axes[0, 1],
                ground_truth_axis=None if j else axes[0, 2],
            )
        axes[1, 0].set_ylabel("Prediction", fontsize=20)
        axes[2, 0].set_ylabel("Uncertainty", fontsize=20)
        for ax in axes.flatten():
            if not ax.has_data():
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            f"samples/gp_scale_{scale:g}_shifted_{shift:g}_ls_{extra['ls']:0.2f}_{i}.png"
        )
        plt.clf()


def _plot(
    name,
    batch,
    output,
    f_pred_axis,
    f_std_axis,
    f_norm=None,
    f_std_norm=None,
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
    kwargs = dict(cmap=cmap, norm=f_norm, interpolation="none")
    kwargs_std = dict(cmap=cmap_std, norm=f_std_norm, interpolation="none")
    f_pred_axis.imshow(f_pred, **kwargs)
    f_std_axis.imshow(f_std, **kwargs_std)
    f_std_axis.set_xlabel(name, fontsize=20)
    if task_axis:
        task_axis.set_title("Task", fontsize=20)
        task_axis.imshow(f_ctx, **kwargs)
    if ground_truth_axis:
        ground_truth_axis.set_title("Ground Truth", fontsize=20)
        ground_truth_axis.imshow(f_test, **kwargs)


def update_convcnp_grid(model, axes: list[DictConfig]):
    state, cfg = model["state"], model["cfg"]
    cfg.model.s_lower = [axis.start - 0.5 for axis in axes]
    cfg.model.s_upper = [axis.stop + 0.5 for axis in axes]
    model = instantiate(cfg.model)
    return {"state": state.replace(apply_fn=model.apply), "cfg": cfg}


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
        "--scale",
        default=4.0,
        type=float,
        help="Number of units in each axis.",
    )
    parser.add_argument(
        "--shift",
        default=0.0,
        type=float,
        help="Shift image by this value.",
    )
    parser.add_argument(
        "--num_ctx",
        default=256,
        type=int,
        help="Number of context points.",
    )
    parser.add_argument(
        "--num_samples",
        default=16,
        type=int,
        help="Number of samples to plot.",
    )
    parser.add_argument(
        "--num_points_per_unit",
        default=16,
        type=int,
        help="Number of points per unit.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    main(parse_args(sys.argv))
