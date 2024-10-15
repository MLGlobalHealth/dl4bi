import re
from glob import glob
from math import prod

import hydra
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from omegaconf import DictConfig

from dl4bi.meta_regression.train_utils import (
    build_gp_dataloader,
    cfg_to_run_name,
    load_ckpt,
    plot_posterior_predictive,
)


@hydra.main(config_name="default", version_base=None)
def main(cfg: DictConfig):
    gp_tasks = re.compile("Gaussian Processes", re.IGNORECASE)
    img_tasks = re.compile("MNIST|CelebA|Cifar", re.IGNORECASE)
    only_regex = re.compile(cfg.get("only", ".*"), re.IGNORECASE)
    exclude_regex = re.compile(cfg.get("exclude", ""), re.IGNORECASE)
    if gp_tasks.match(cfg.project):
        if "1d" in cfg.data.name:
            return plot_1d_gp_samples(cfg, only_regex, exclude_regex)
        return plot_2d_gp_samples(cfg, only_regex, exclude_regex)
    elif img_tasks.match(cfg.project):
        return plot_2d_img_samples(cfg, only_regex, exclude_regex)


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
    ckpt = {}
    for p in glob(dir + "*.ckpt"):
        if only_regex.match(str(p)) and not exclude_regex.match(str(p)):
            state, tmp_cfg = load_ckpt(p)
            ckpt[cfg_to_run_name(tmp_cfg)] = state
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


def plot_2d_img_samples(rng, cfg, **kwargs):
    pass


if __name__ == "__main__":
    main()
