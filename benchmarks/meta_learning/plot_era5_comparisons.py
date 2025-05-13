#!/usr/bin/env python3
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from era5 import build_dataloaders, t_to_label
from jax import jit, random
from tqdm import tqdm

from dl4bi.core.train import load_ckpt
from dl4bi.core.utils import nan_pad
from dl4bi.meta_learning.data.spatiotemporal import _inv_permute_Ls
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
    plot(models, args.num_ctx_per_t, args.num_samples)


def plot(
    models,
    num_ctx_per_t: int = 32,
    num_samples: int = 16,
):
    key = list(models.keys())[0]
    cfg = models[key]["cfg"]  # cfg.data should be the same for all
    cfg.data.batch_size = 1
    cfg.data.num_ctx_per_t = num_ctx_per_t
    cfg.data.num_ctx_per_t = num_ctx_per_t
    *_, dataloader = build_dataloaders(cfg.data, cfg.kernel)
    rng = random.key(cfg.seed)
    rng_data, rng = random.split(rng)
    batches = dataloader(rng_data)
    num_models = len(models)
    Path("samples").mkdir(exist_ok=True)
    for i in tqdm(range(1, num_samples + 1)):
        rng_i, rng = random.split(rng)
        batch, revert = next(batches)
        batch = replace(
            batch,
            s_ctx=revert["s"](batch.s_ctx),
            s_test=revert["s"](batch.s_test),
            t_ctx=t_to_label(revert["t"](batch.t_ctx)),
            t_test=t_to_label(revert["t"](batch.t_test)),
        )
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
        fig, axes = plt.subplots(3, len(models), figsize=(3 * num_models, 9))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        for j, (model_cls_name, d) in enumerate(models.items()):
            name = {
                "BSATNP": "BSA-TNP",
                "TNPD": "TNP-D",
                "TETNP": "PT-TE-TNP",
            }
            (T_b, L), (B, _, D_f) = batch.inv_permute_idx.shape, batch.f_ctx.shape
            # fill in masked values with nan
            f_ctx = jnp.where(batch.mask_ctx[..., None], batch.f_ctx, jnp.nan)
            f_test = batch.f_test
            # reintroduce timestep and nan pad each time step to full size
            f_ctx = f_ctx.reshape(B, T_b - 1, -1, D_f)
            f_ctx = nan_pad(f_ctx, axis=2, L=L)
            f_test, f_pred, f_std = unbatch_BLD([f_test, f_pred, f_std], L)
            f_test, f_pred, f_std = map(lambda v: v[:, None], [f_test, f_pred, f_std])
            # invert permutation of the flattened spatial dim, L, by time step
            _, f_test = _inv_permute_Ls(f_ctx, f_test, batch.inv_permute_idx)
            _, f_pred = _inv_permute_Ls(f_ctx, f_pred, batch.inv_permute_idx)
            f_ctx, f_std = _inv_permute_Ls(f_ctx, f_std, batch.inv_permute_idx)
            # reshape to original spatial dims
            reshape_s = jit(
                lambda v: v.reshape(*v.shape[:2], *batch.s_dims, v.shape[-1])
            )
            f_ctx, f_test, f_pred, f_std = map(
                reshape_s, [f_ctx, f_test, f_pred, f_std]
            )
            cmap = mpl.colormaps.get_cmap("Spectral_r")
            cmap.set_bad("grey")
            kwargs = dict(cmap=cmap, norm=norm, interpolation="none")
            std_kwargs = dict(cmap="plasma", norm=norm_std, interpolation="none")
        plt.tight_layout()
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
