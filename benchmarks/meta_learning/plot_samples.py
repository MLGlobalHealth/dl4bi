#!/usr/bin/env python3
import re
from pathlib import Path

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
from celeba import build_dataloaders as build_dataloaders_celeba
from cifar_10 import build_dataloaders as build_dataloaders_cifar_10
from gp import (
    build_2d_grid_gp_dataloader,
    build_gp_dataloader,
    plot_posterior_predictive,
)
from jax import random
from mnist import build_dataloaders as build_dataloaders_mnist
from omegaconf import DictConfig
from sir import build_dataloader as build_dataloader_sir
from sir import remap_colors as remap_colors_sir

from dl4bi.meta_learning.train_utils import (
    TrainState,
    instantiate,
    load_ckpts,
    plot_img,
    regression_to_rgb,
)
from dl4bi.meta_learning.transform import pointwise_multinomial


@hydra.main(config_name="default", version_base=None)
def main(cfg: DictConfig):
    gp_tasks = re.compile(".*Gaussian Processes.*", re.IGNORECASE)
    img_tasks = re.compile(".*(MNIST|CelebA|Cifar|SIR).*", re.IGNORECASE)
    only_regex = re.compile(cfg.get("only", ".*"), re.IGNORECASE)
    exclude_regex = re.compile(cfg.get("exclude", r"^$"), re.IGNORECASE)
    num_ctx = cfg.get("num_ctx", 10)
    num_samples = cfg.get("num_samples", 16)
    results_dir = Path(f"results/{cfg.project}")
    if gp_tasks.match(cfg.project):
        results_dir /= f"{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}"
        ckpts = load_ckpts(results_dir, only_regex, exclude_regex)
        if "1d" in cfg.data.name:
            return plot_1d_gp_samples(cfg, ckpts, num_ctx, num_samples, results_dir)
        return plot_2d_img_samples(cfg, ckpts, num_ctx, num_samples, results_dir)
    elif img_tasks.match(cfg.project):
        results_dir /= f"{cfg.seed}"
        ckpts = load_ckpts(results_dir, only_regex, exclude_regex)
        return plot_2d_img_samples(cfg, ckpts, num_ctx, num_samples, results_dir)


def plot_1d_gp_samples(
    cfg: DictConfig,
    ckpts: dict,
    num_ctx: int = 10,
    num_samples: int = 16,
    results_dir: Path = Path("."),
    lengthscales: list[float] = [0.1, 0.3, 0.5],
):
    rng = random.key(cfg.seed)
    rng_data, rng_dropout, rng_extra = random.split(rng, 3)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = num_ctx
    cfg.data.num_ctx.max = num_ctx
    cfg.kernel.kwargs.ls.kwargs.dist = "fixed"
    num_rows, num_cols = len(ckpts), len(lengthscales)
    fig_size = (6 * num_cols, 4 * num_rows)
    for i in range(num_samples):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
        rng_i, rng_data = random.split(rng_data)
        for col_idx, ls in enumerate(lengthscales):
            cfg.kernel.kwargs.ls.kwargs.kwargs = {"value": ls}
            dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *_ = next(dataloader(rng_i))
            for row_idx, run_name in enumerate(sorted(ckpts)):
                state = ckpts[run_name]["state"]
                output = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_ctx,
                    f_ctx,
                    s_test,
                    valid_lens_ctx,
                    rngs={"dropout": rng_dropout, "extra": rng_extra},
                )
                if isinstance(output[1], tuple):  # latent or bootsrapping
                    output, _ = output  # throw away latent or base samples
                f_mu, f_std = output
                plt.sca(axs[row_idx, col_idx])
                plot_posterior_predictive(
                    s_ctx[0, :num_ctx, 0],
                    f_ctx[0, :num_ctx, 0],
                    s_test[0, :, 0],
                    f_test[0, :, 0],
                    f_mu[..., 0],
                    f_std[..., 0],
                )
                # NOTE: only set labels for non-interior plots
                axs[row_idx, col_idx].set_xticks([])
                axs[row_idx, col_idx].set_yticks([])
                axs[row_idx, col_idx].set_xlabel("")
                axs[row_idx, col_idx].set_ylabel("")
                if row_idx == 0:
                    axs[row_idx, col_idx].set_title(
                        f"lengthscale={ls:0.1f}", fontsize=36
                    )
                if col_idx == 0:
                    short_name = shortened_run_name(run_name)
                    axs[row_idx, col_idx].set_ylabel(short_name, fontsize=36)
        fig.tight_layout()
        fig.savefig(results_dir / f"comparison_sample_{i + 1}.png", dpi=150)
        plt.clf()


def shortened_run_name(name):
    match name:
        case "Scan TNP-KR: TISABiasedScanAttention":
            return "TNP-KR: SA"
        case "TNP-KR: DeepKernelAttention":
            return "TNP-KR: DKA"
        case "TNP-KR: FastAttention":
            return "TNP-KR: PERF"
        case _:
            return name


def plot_2d_img_samples(
    cfg: DictConfig,
    ckpts: dict,
    num_ctx: int = 50,
    num_samples: int = 16,
    results_dir: Path = Path("."),
):
    build_dataloaders, shape, cmap, cmap_std, remap_colors, is_categorical = (
        project_parameters(cfg)
    )
    H, W, C = shape
    if build_dataloaders == build_2d_grid_gp_dataloader:
        cfg.data.update({"batch_size": 1, "num_ctx": {"max": num_ctx, "min": num_ctx}})
        train_dataloader = build_dataloaders(cfg.data, cfg.kernel)
    elif build_dataloaders == build_dataloader_sir:
        cfg.data.update({"batch_size": 1, "num_ctx": {"max": num_ctx, "min": num_ctx}})
        train_dataloader = build_dataloaders(cfg.data, cfg.sim)
    else:
        train_dataloader, *_ = build_dataloaders(
            batch_size=1,
            num_ctx_min=num_ctx,
            num_ctx_max=num_ctx,
            num_test_max=H * W,
        )
    rng = random.key(cfg.seed)
    num_models = len(ckpts)
    for i in range(num_samples):
        rng_dropout, rng_extra, rng_i, rng = random.split(rng, 4)
        (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            _,
            _,
            valid_lens_test,
            s_test_full,
            f_test_full,
            inv_permute_idx,
        ) = next(train_dataloader(rng_i))
        num_rows, num_cols = num_models, 4
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(6 * num_cols, 6 * num_rows),
        )
        preds = {}
        min_std, max_std = float("inf"), -float("inf")
        for row_idx, run_name in enumerate(sorted(ckpts)):
            state = ckpts[run_name]["state"]
            if run_name == "ConvCNP":
                state = update_convcnp_state(ckpts[run_name], shape)
            output = state.apply_fn(
                {"params": state.params, **state.kwargs},
                s_ctx,
                f_ctx,
                s_test_full,
                valid_lens_ctx,
                valid_lens_test,
                rngs={"dropout": rng_dropout, "extra": rng_extra},
            )
            if isinstance(output[1], tuple):  # latent
                output, _ = output  # throw away latent samples
            if is_categorical:
                output = pointwise_multinomial(output)
            f_mu, f_std = output
            preds[run_name] = (f_mu, f_std)
            min_std = min(min_std, f_std.min())
            max_std = max(max_std, f_std.max())
        norm_std = mpl.colors.Normalize(vmin=min_std, vmax=max_std)
        for row_idx, run_name in enumerate(sorted(ckpts)):
            f_mu, f_std = preds[run_name]
            plot_img(
                i,
                shape,
                f_ctx[0, :num_ctx],
                f_mu[0],
                f_std[0],
                f_test_full[0],
                inv_permute_idx,
                axs[row_idx],
                cmap=cmap,
                cmap_std=cmap_std,
                norm_std=norm_std,
                remap_colors=remap_colors,
            )
            # NOTE: unset titles by row, unless its the first row
            for col_idx in range(num_cols):
                axs[row_idx, col_idx].set_xticks([])
                axs[row_idx, col_idx].set_yticks([])
                if row_idx != 0:
                    axs[row_idx, col_idx].set_title("")
                else:
                    title = axs[row_idx, col_idx].get_title()
                    axs[row_idx, col_idx].set_title(title, fontsize=36)
            short_name = shortened_run_name(run_name)
            axs[row_idx, 0].set_ylabel(f"{short_name}", fontsize=36)
        fig.tight_layout()
        fig.savefig(results_dir / f"comparison_sample_{i + 1}.png", dpi=150)
        plt.clf()


def project_parameters(cfg: DictConfig):
    cmap = mpl.colormaps.get_cmap("Spectral_r")
    cmap_std = mpl.colormaps.get_cmap("plasma")
    cmap.set_bad("grey")
    shape = (32, 32, 3)
    # example project names include "TNP-KR - MNIST", "MNIST", etc, so match
    matches = lambda pattern: re.match(pattern, cfg.project, re.IGNORECASE)
    remap_colors = lambda x: x
    is_categorical = False
    match cfg.project:
        case _ if matches(".*Gaussian Processes.*"):
            build_dataloaders = build_2d_grid_gp_dataloader
            shape = (16, 16, 1)
            cmap = mpl.colormaps.get_cmap("grey")
            cmap.set_bad("blue")
        case _ if matches(".*MNIST.*"):
            build_dataloaders = build_dataloaders_mnist
            shape = (28, 28, 1)
            cmap = mpl.colormaps.get_cmap("grey")
            cmap.set_bad("blue")
        case _ if matches(".*CelebA.*"):
            build_dataloaders = build_dataloaders_celeba
            remap_colors = regression_to_rgb
        case _ if matches(".*Cifar.*"):
            build_dataloaders = build_dataloaders_cifar_10
            remap_colors = regression_to_rgb
        case _ if matches(".*SIR.*"):
            build_dataloaders = build_dataloader_sir
            remap_colors = remap_colors_sir
            shape = {
                "space_64x64": (64, 64, 3),
                "space_128x128": (128, 128, 3),
                "space_256x256": (256, 256, 3),
                "space_1024x1024": (1024, 1024, 3),
            }[cfg.data.name]
            is_categorical = True
        case _:
            raise Exception(f"No dataloader defined for {cfg.project}!")
    return build_dataloaders, shape, cmap, cmap_std, remap_colors, is_categorical


def update_convcnp_state(ckpt, shape):
    state, cfg = ckpt["state"], ckpt["cfg"]
    if shape[0] == 128:
        cfg.model.kwargs.s_lower = [-4.5, -4.5]
        cfg.model.kwargs.s_upper = [4.5, 4.5]
    elif shape[0] == 256:
        cfg.model.kwargs.s_lower = [-8.5, -8.5]
        cfg.model.kwargs.s_upper = [8.5, 8.5]
    elif shape[0] == 1024:
        cfg.model.kwargs.s_lower = [-32.5, -32.5]
        cfg.model.kwargs.s_upper = [32.5, 32.5]
    model = instantiate(cfg.model)
    print(model)
    state = TrainState.create(
        apply_fn=model.apply,
        params=state.params,
        kwargs=state.kwargs,
        tx=state.tx,
    )
    return state


if __name__ == "__main__":
    main()
