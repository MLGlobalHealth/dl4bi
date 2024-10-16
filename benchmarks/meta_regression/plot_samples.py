import re
from glob import glob
from math import prod

import hydra
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from celeba import build_dataloaders as build_dataloaders_celeba
from cifar_10 import build_dataloaders as build_dataloaders_cifar_10
from jax import random
from mnist import build_dataloaders as build_dataloaders_mnist
from omegaconf import DictConfig

from dl4bi.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
    plot_img,
    plot_posterior_predictive,
)


def get_ckpts(dir: str, only_regex: re.Pattern, exclude_regex: re.Pattern):
    ckpt = {}
    for p in glob(dir + "*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = state
    return ckpt


@hydra.main(config_name="default", version_base=None)
def main(cfg: DictConfig):
    gp_tasks = re.compile("Gaussian Processes", re.IGNORECASE)
    img_tasks = re.compile("MNIST|CelebA|Cifar", re.IGNORECASE)
    only_regex = re.compile(cfg.get("only", ".*"), re.IGNORECASE)
    exclude_regex = re.compile(cfg.get("exclude", r"a^"), re.IGNORECASE)
    num_samples = cfg.get("num_samples", None)
    if gp_tasks.match(cfg.project):
        if "1d" in cfg.data.name:
            return plot_1d_gp_samples(cfg, only_regex, exclude_regex)
        return plot_2d_gp_samples(cfg, only_regex, exclude_regex)
    elif img_tasks.match(cfg.project):
        return plot_2d_img_samples(cfg, only_regex, exclude_regex, num_samples)


def plot_1d_gp_samples(
    cfg,
    only_regex,
    exclude_regex,
    num_ctx=10,
    lengthscales=[0.1, 0.3, 0.5],
    num_samples=16,
):
    rng = random.key(cfg.seed)
    rng_data, rng_dropout, rng_extra = random.split(rng, 3)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = num_ctx
    cfg.data.num_ctx.max = num_ctx
    cfg.kernel.kwargs.ls.kwargs.dist = "fixed"
    dir = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/"
    ckpt = get_ckpts(dir, only_regex, exclude_regex)
    num_rows, num_cols = len(ckpt), len(lengthscales)
    fig_size = (6 * num_cols, 4 * num_rows)
    for i in range(num_samples):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
        rng_i, rng_data = random.split(rng_data)
        for col, ls in enumerate(lengthscales):
            cfg.kernel.kwargs.ls.kwargs.kwargs = {"value": ls}
            dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
            s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, *_ = next(dataloader(rng_i))
            for row, run_name in enumerate(ckpt):
                state = ckpt[run_name]
                f_mu, f_std, *_ = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_ctx,
                    f_ctx,
                    s_test,
                    valid_lens_ctx,
                    rngs={"dropout": rng_dropout, "extra": rng_extra},
                )
                plt.sca(axs[row, col])
                plot_posterior_predictive(
                    s_ctx[0, :num_ctx, 0],
                    f_ctx[0, :num_ctx, 0],
                    s_test[0, :, 0],
                    f_test[0, :, 0],
                    f_mu[..., 0],
                    f_std[..., 0],
                )
                if row == 0:
                    axs[row, col].set_title(f"ls={ls:0.1f}")
                if col == 0:
                    axs[row, col].set_ylabel(run_name)
                if row == num_rows - 1:
                    axs[row, col].set_xlabel("s")
        fig.tight_layout()
        fig.savefig(dir + f"comparison_sample_{i+1}.png", dpi=150)
        plt.clf()


def plot_2d_gp_samples(
    cfg,
    only_regex,
    exclude_regex,
    num_ctx=128,
    num_samples=16,
):
    rng = random.key(cfg.seed)
    cfg.data.batch_size = 1
    cfg.data.num_ctx.min = 0  # use only grid points in dataloader
    cfg.data.num_ctx.max = 0
    cfg.kernel.kwargs.ls.kwargs.dist = "fixed"
    dir = f"results/gp/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/"
    ckpt = {}
    for p in glob(dir + "*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = state
    num_rows, num_cols = len(ckpt), 4
    fig_size = (6 * num_cols, 4 * num_rows)
    L = prod([axis.num for axis in cfg.data.s])
    for i in range(num_samples):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=fig_size)
        rng_data, rng_permute, rng_dropout, rng_extra, rng = random.split(rng, 5)
        for col in range(num_cols):
            dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
            _, _, _, s, f, *_ = next(dataloader(rng_data))
            permute_idx = random.choice(rng_permute, L, (L,), replace=False)
            inv_permute_idx = jnp.argsort(permute_idx)
            # permute the order and select the first valid_lens_ctx for context
            s_permuted = s[:, permute_idx, :]
            f_permuted = f[:, permute_idx, :]
            valid_lens_ctx = jnp.array([num_ctx])
            for row, run_name in enumerate(ckpt):
                state = ckpt[run_name]
                f_mu, f_std, *_ = state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    s_permuted,
                    f_permuted,
                    s_permuted,
                    valid_lens_ctx,
                    rngs={"dropout": rng_dropout, "extra": rng_extra},
                )
                plt.sca(axs[row, col])
                plt.clf()
                # TODO(danj): PLOT
        fig.tight_layout()
        fig.savefig(dir + f"comparison_sample_{i+1}.png", dpi=150)
        plt.clf()


def get_task_image_dims(task: str):
    channels = 3 if task != "MNIST" else 1
    img_ax_size = 32 if task != "MNIST" else 28
    return channels, img_ax_size


def get_image_task_dataloader(cfg: DictConfig, num_ctx: int, img_ax_size: int):
    L = img_ax_size * img_ax_size
    build_data_loader = {
        "MNIST": build_dataloaders_mnist,
        "CelebA": build_dataloaders_celeba,
        "Cifar_10": build_dataloaders_cifar_10,
    }[cfg.project]
    return build_data_loader(
        batch_size=1, num_ctx_max=num_ctx, num_ctx_min=num_ctx, num_test_max=L
    )[-1]


def plot_2d_img_samples(cfg, only_regex, exclude_regex, num_samples=16, num_ctx=50):
    dir = f"results/{cfg.project.lower()}/{cfg.seed}/"
    ckpt = get_ckpts(dir, only_regex, exclude_regex)
    channels, img_ax_size = get_task_image_dims(cfg.project)
    task_dataloader = get_image_task_dataloader(cfg, num_ctx, img_ax_size)
    for example in range(num_samples):
        min_boundary, max_boundary = 10, -10
        rng = random.key(example)
        rng_extra, rng_sample = random.split(rng, 2)
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
        ) = next(task_dataloader(rng_sample))
        plt.clf()
        num_rows, num_cols = len(ckpt), 4
        fig, axs = plt.subplots(
            num_rows,
            num_cols,
            figsize=(9 * num_cols, 6 * num_rows),
        )

        model_preds = {}
        for row_idx, model_name in enumerate(ckpt):
            state = ckpt[model_name]

            f_mu, f_std, *_ = state.apply_fn(
                {"params": state.params, **state.kwargs},
                s_ctx,
                f_ctx,
                s_test_full,
                valid_lens_ctx,
                valid_lens_test,
                rngs={"extra": rng_extra},
            )
            model_preds[model_name] = (f_mu, f_std)
            min_boundary = min(min_boundary, f_std.min())
            max_boundary = max(max_boundary, f_std.max())

        sm = mpl.cm.ScalarMappable(
            cmap="Spectral_r",
            norm=mpl.colors.Normalize(vmin=min_boundary, vmax=max_boundary),
        )
        sm.set_array([])

        for row_idx, model_name in enumerate(ckpt):
            f_mu, f_std = model_preds[model_name]

            plot_img(
                0,
                (img_ax_size, img_ax_size, channels),
                f_ctx[0, :num_ctx],
                f_mu[0],
                f_std[0],
                f_test_full[0],
                inv_permute_idx,
                axs[row_idx],
            )

            for col_idx in range(num_cols):
                if row_idx != 0:
                    axs[row_idx, col_idx].set_title("")
                else:
                    axs[row_idx, col_idx].set_title(
                        axs[row_idx, col_idx].get_title(), fontsize=20
                    )

            axs[row_idx, 0].set_ylabel(f"{model_name}", fontsize=20)

        cbar_ax = fig.add_axes([0.05, 0.3, 0.02, 0.4])
        fig.colorbar(sm, cax=cbar_ax)

        fig.tight_layout(rect=[0.05, 0, 1, 1])
        fig.savefig(dir + f"comparison_sample_{example+1}.png", dpi=150)
        plt.clf()


if __name__ == "__main__":
    main()
