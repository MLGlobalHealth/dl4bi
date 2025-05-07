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
from beijing_air_quality import main as beijing_air_quality_main
from celeba import main as celeba_main
from cifar_10 import main as cifar_10_main
from era5 import main as era5_main
from gp import main as gp_main
from household_electric import main as household_electric_main
from hydra import compose, initialize
from jax import random
from mnist import main as mnist_main
from sir import main as sir_main


def b_tnp_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the Transformer Neural Process - Kernel Regresssion (BSA-TNP) paper."""
    overrides = []
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides = [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]
    gp_kernels_2d = ["rbf"]
    models = [
        "bsa_tnp",
        "tnp_d",
        "te_tnp",
        "convcnp",
    ]

    # TRANSLATION INVARIANCE
    # gp_benchmark(
    #     seeds,
    #     "2d",
    #     gp_kernels_2d,
    #     [f"2d/{m}" for m in models],
    #     gp_main,
    #     overrides,
    #     "NeurIPS BSA-TNP - Gaussian Processes",
    #     dry_run=dry_run,
    # )
    # generic_benchmark(
    #     seeds,
    #     "configs/sir",
    #     models,
    #     sir_main,
    #     overrides,
    #     "NeurIPS BSA-TNP - SIR",
    #     dry_run=dry_run,
    # )
    # TODO(danj): look in plot_samples
    # test_translation(...)
    # test_multiresolution(...)

    # SPACE & TIME & GENERALIZATION
    # era5_models = ["tnp_d", "te_tnp", "bsa_tnp"]
    # era5_overrides = [
    #     "data.valid_region=northern_europe",
    #     "data.test_region=western_europe",
    # ]
    # generic_benchmark(
    #     seeds,
    #     "configs/era5",
    #     era5_models,
    #     era5_main,
    #     overrides + era5_overrides,
    #     "NeurIPS BSA-TNP - ERA5 - CNW",
    #     dry_run=dry_run,
    # )
    # era5_overrides = [
    #     "data.valid_region=western_europe",
    #     "data.test_region=northern_europe",
    # ]
    # generic_benchmark(
    #     seeds,
    #     "configs/era5",
    #     era5_models,
    #     era5_main,
    #     overrides + era5_overrides,
    #     "NeurIPS BSA-TNP - ERA5 - CWN",
    #     dry_run=dry_run,
    # )

    # HIGH DIMENSIONAL
    # TODO(danj):
    # fixed + spatial effects:
    tabular_models = ["bsa_tnp", "te_tnp"]
    generic_benchmark(
        seeds,
        "configs/household_electric",
        ["bsa_tnp"],  # add temporal bias
        # tabular_models,
        household_electric_main,
        overrides,
        "NeurIPS BSA-TNP - Household Electric",
        dry_run=dry_run,
    )
    generic_benchmark(
        seeds,
        "configs/beijing_air_quality",
        tabular_models,
        beijing_air_quality_main,
        overrides,
        "NeurIPS BSA-TNP - Beijing Air Quality",
        dry_run=dry_run,
    )

    # NON-STATIONARY DISTRIBUTIONS
    # generic_benchmark(
    #     seeds,
    #     "configs/celeba",
    #     models,
    #     celeba_main,
    #     overrides,
    #     "NeurIPS BSA-TNP - CelebA",
    #     dry_run=dry_run,
    # )
    # generic_benchmark(
    #     seeds,
    #     "configs/cifar_10",
    #     models,
    #     cifar_10_main,
    #     overrides,
    #     "NeurIPS BSA-TNP - Cifar 10",
    #     dry_run=dry_run,
    # )


def gp_benchmark(
    seeds: jax.Array,
    data: str = "1d",
    kernels: list[str] = ["rbf", "periodic", "b_tnp"],
    models: list[str] = ["1d/bsa_tnp"],
    main_fn: Callable = gp_main,
    overrides: list = [],
    project: str = "",
    project_parent: str = "None",
    dry_run: bool = False,
):
    if dry_run:
        project = "__DRY RUN__ " + project
        project_parent = "__DRY RUN__ " + project_parent
    for kernel in kernels:
        for seed in seeds:
            for model in models:
                with initialize(config_path="configs/gp", version_base=None):
                    cfg = compose(
                        "default",
                        overrides=[
                            f"project={project}",
                            f"data={data}",
                            f"model={model}",
                            f"kernel={kernel}",
                            f"seed={seed}",
                            f"+project_parent={project_parent}",
                        ]
                        + overrides,
                    )
                    print("=" * 100)
                    main_fn(cfg)


def generic_benchmark(
    seeds: jax.Array,
    cfg_dir: str = "configs/celeba",
    models: list[str] = ["bsa_tnp"],
    main_fn: Callable = celeba_main,
    overrides: list = [],
    project: str = "",
    dry_run: bool = False,
):
    if dry_run:
        project = "__DRY RUN__ " + project
    for seed in seeds:
        for model in models:
            with initialize(config_path=cfg_dir, version_base=None):
                cfg = compose(
                    "default",
                    overrides=[
                        f"project={project}",
                        f"model={model}",
                        f"seed={seed}",
                    ]
                    + overrides,
                )
                print("=" * 100)
                main_fn(cfg)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=88,
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
    rng = random.key(args.seed)
    seeds = random.choice(rng, 100, (args.num_runs,), replace=False)
    b_tnp_paper(seeds, args.dry_run)
