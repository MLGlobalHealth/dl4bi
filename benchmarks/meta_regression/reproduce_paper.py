#!/usr/bin/env python3
"""
This file emulates calling all the scripts as if they were called from CLI.

*This code is instrumented with logging using Weights & Biases (wandb.ai),
so an account is necessary to log and collect the results from these runs.
"""

import argparse
import sys
from collections.abc import Callable

import jax
from bayes_opt import main as bayes_opt_main
from celeba import main as celeba_main
from gp import main as gp_main
from hydra import compose, initialize
from jax import random
from mnist import main as mnist_main


def sptx_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the SPTx: Stochastic Process Transformer."""
    overrides = []
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides = [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]
    overrides = [overrides]  # expects list[list[str]]
    gp_kernels_1d = ["periodic", "rbf", "matern_3_2"]
    gp_kernels_2d = ["rbf"]
    models = [
        "tnpd",
        "sptx_full_rff",
        "sptx_fast_rff",
        "sptx_full",
        "sptx_fast",
        "np",
        "bnp",
        "cnp",
        "anp",
        "canp",
        "banp",
        "convcnp",
    ]
    exclude_2d = ["bnp", "banp", "convcnp"]
    models_2d = [m for m in models if m not in exclude_2d]
    gp_benchmark(seeds, 1, gp_kernels_1d, models, gp_main, overrides)
    gp_benchmark(seeds, 1, gp_kernels_1d, models, bayes_opt_main, overrides)
    gp_benchmark(seeds, 2, gp_kernels_2d, models_2d, gp_main, overrides)
    img_benchmark(seeds, "configs/mnist", models_2d, mnist_main, overrides)
    img_benchmark(seeds, "configs/celeba", models_2d, celeba_main, overrides)
    # TODO(danj): add MACE by Lengthscale plots


def lore_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the LoRe: Local Refinement with Tranformers paper."""
    gp_kernels_1d = ["rbf", "periodic", "matern_3_2"]
    models = ["sptx_full"]
    overrides = []
    for num_blks, num_reps in [(6, 1), (3, 2), (2, 3), (1, 6)]:
        blk_str = f"model.kwargs.dec.kwargs.num_blks={num_blks}"
        rep_str = f"model.kwargs.dec.kwargs.num_reps={num_reps}"
        overrides += [[blk_str, rep_str]]
    gp_benchmark(seeds, 1, gp_kernels_1d, models, gp_main, overrides)
    img_benchmark(seeds, "configs/mnist", models, mnist_main, overrides)
    img_benchmark(seeds, "configs/celeba", models, celeba_main, overrides)
    # TODO(danj): GPT-2 LLM benchmark: [(12, 1), (6, 2), (4, 3), (3, 4)]


def gp_benchmark(
    seeds: jax.Array,
    dim: int = 1,
    kernels: list[str] = ["rbf", "periodic", "sptx"],
    models: list[str] = ["sptx_fast"],
    main_fn: Callable = gp_main,
    overrides: list[list[str]] = [[]],
):
    for kernel in kernels:
        for seed in seeds:
            for model in models:
                for _overrides in overrides:
                    with initialize(config_path="configs/gp", version_base=None):
                        cfg = compose(
                            "default",
                            overrides=[
                                f"data={dim}d",
                                f"model={model}",
                                f"kernel={kernel}",
                                f"seed={seed}",
                            ]
                            + _overrides,
                        )
                        print("=" * 100)
                        main_fn(cfg)


def img_benchmark(
    seeds: jax.Array,
    cfg_dir: str = "configs/mnist",
    models: list[str] = ["sptx_fast"],
    main_fn: Callable = mnist_main,
    overrides: list[list[str]] = [[]],
):
    for seed in seeds:
        for model in models:
            for _overrides in overrides:
                with initialize(config_path=cfg_dir, version_base=None):
                    cfg = compose(
                        "default",
                        overrides=[
                            f"model={model}",
                            f"seed={seed}",
                        ]
                        + _overrides,
                    )
                    print("=" * 100)
                    main_fn(cfg)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paper",
        choices=["sptx", "lore"],
        default="sptx",
        help="Select which paper to reproduce.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=7,
        help="One seed to rule (seed) them all.",
    )
    parser.add_argument(
        "-n",
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for each reported value.",
    )
    parser.add_argument(
        "-d",
        "--dry_run",
        action="store_true",
        help="A complete dry run without wandb logging.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    seeds = random.randint(random.key(args.seed), (args.num_runs,), 0, 100)
    match args.paper:
        case "sptx":
            sptx_paper(seeds, args.dry_run)
        case "lore":
            lore_paper(seeds, args.dry_run)
        case _:
            pass
