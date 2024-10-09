#!/usr/bin/env python3
"""
This file emulates calling all the scripts as if they were called from CLI.

*This code is instrumented with logging using Weights & Biases (wandb.ai),
so an account is necessary to log and collect the results from these runs.
"""

import argparse
import itertools as it
import sys
from collections.abc import Callable

import jax
from bayes_opt import main as bayes_opt_main
from celeba import main as celeba_main
from gp import main as gp_main
from hydra import compose, initialize
from jax import random
from mnist import main as mnist_main


def tnp_kr_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the Transformer Neural Process - Kernel Regresssion (TNP-KR) paper."""
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
        "tnp_d",
        "tnp_kr_full_rff",
        "tnp_kr_fast_rff",
        "tnp_kr_full",
        "tnp_kr_fast",
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
    gp_models_2d = [m.replace("rff", "rff_2d") for m in models_2d]
    gp_benchmark(
        seeds,
        1,
        gp_kernels_1d,
        models,
        gp_main,
        overrides,
        "TNP-KR - Gaussian Processes",
    )
    gp_benchmark(
        seeds,
        1,
        gp_kernels_1d,
        models,
        bayes_opt_main,
        overrides,
        "TNP-KR - Bayesian Optimization",
    )
    # gp_benchmark(
    #     seeds,
    #     2,
    #     gp_kernels_2d,
    #     gp_models_2d,
    #     gp_main,
    #     overrides,
    #     "TNP-KR - Gaussian Processes",
    # )
    # img_benchmark(
    #     seeds,
    #     "configs/mnist",
    #     models_2d,
    #     mnist_main,
    #     overrides,
    #     "TNP-KR - MNIST",
    # )
    # img_benchmark(
    #     seeds,
    #     "configs/celeba",
    #     models_2d,
    #     celeba_main,
    #     overrides,
    #     "TNP-KR - CelebA",
    # )


def lore_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the LoRe: Local Refinement with Tranformers paper."""
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
    gp_kernels_1d = ["rbf", "periodic", "matern_3_2"]
    gp_kernels_2d = ["rbf"]
    models = ["tnp_kr_full"]
    overrides = []
    nums = [1, 2, 3, 6]
    for num_blks, num_reps in it.product(nums, nums):
        blk_str = f"model.kwargs.dec.kwargs.num_blks={num_blks}"
        rep_str = f"model.kwargs.dec.kwargs.num_reps={num_reps}"
        override = [blk_str, rep_str]
        if dry_run:
            override += [
                "wandb=False",
                "train_num_steps=100",
                "valid_num_steps=50",
                "plot_interval=50",
            ]
        overrides += [override]
    gp_benchmark(
        seeds,
        1,
        gp_kernels_1d,
        models,
        gp_main,
        overrides,
        "LoRe - Gaussian Processes",
    )
    gp_benchmark(
        seeds,
        2,
        gp_kernels_2d,
        models,
        gp_main,
        overrides,
        "LoRe - Gaussian Processes",
    )
    img_benchmark(
        seeds,
        "configs/mnist",
        models,
        mnist_main,
        overrides,
        "LoRe - MNIST",
    )
    img_benchmark(
        seeds,
        "configs/celeba",
        models,
        celeba_main,
        overrides,
        "LoRe - CelebA",
    )
    # TODO(danj): GPT-2 LLM benchmark: [(12, 1), (6, 2), (6, 1)]


def gp_benchmark(
    seeds: jax.Array,
    dim: int = 1,
    kernels: list[str] = ["rbf", "periodic", "tnp_kr"],
    models: list[str] = ["tnp_kr_fast"],
    main_fn: Callable = gp_main,
    overrides: list[list[str]] = [[]],
    project: str = "",
):
    for kernel in kernels:
        for seed in seeds:
            for model in models:
                for _overrides in overrides:
                    with initialize(config_path="configs/gp", version_base=None):
                        cfg = compose(
                            "default",
                            overrides=[
                                f"project={project}",
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
    models: list[str] = ["tnp_kr_fast"],
    main_fn: Callable = mnist_main,
    overrides: list[list[str]] = [[]],
    project: str = "",
):
    for seed in seeds:
        for model in models:
            for _overrides in overrides:
                with initialize(config_path=cfg_dir, version_base=None):
                    cfg = compose(
                        "default",
                        overrides=[
                            f"project={project}",
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
        choices=["tnp_kr", "lore"],
        default="tnp_kr",
        help="Select which paper to reproduce.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=8,
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
        case "tnp_kr":
            tnp_kr_paper(seeds, args.dry_run)
        case "lore":
            lore_paper(seeds, args.dry_run)
        case _:
            pass
