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
from era5 import main as era5_main
from gneiting_gp import main as gneiting_gp_main
from gp import main as gp_main
from hydra import compose, initialize
from jax import random
from multiscale_2d_gp import main as multiscale_2d_gp_main
from sir import main as sir_main


def bsa_tnp_paper(seeds: jax.Array, dry_run: bool = False):
    """Reproduces the BSA-TNP paper."""
    overrides = []
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides = [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "++test_num_steps=50",
            "++plot_interval=50",
        ]
    gp_translation_invariance_benchmarks(seeds, overrides, dry_run)
    gp_multi_scale_benchmarks(seeds, overrides, dry_run)
    gp_rotational_invariance_benchmarks(seeds, overrides, dry_run)
    sir_benchmarks(seeds, overrides, dry_run)
    era5_benchmarks(seeds, overrides, dry_run)
    beijing_benchmarks(seeds, overrides, dry_run)
    gneiting_gp_benchmarks(seeds, overrides, dry_run)


def gp_translation_invariance_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    gp_kernels_2d = ["rbf"]
    models = [
        "bsa_tnp",
        "tnp_d",
        "te_tnp",
        "convcnp",
    ]
    gp_benchmark(
        seeds,
        "2d",
        gp_kernels_2d,
        [f"2d/{m}" for m in models],
        gp_main,
        overrides,
        "AISTATS BSA-TNP - Gaussian Processes",
        dry_run=dry_run,
    )
    # as requested by reviewer
    gp_benchmark(
        seeds,
        "2d",
        gp_kernels_2d,
        [f"2d/{m}" for m in ["te_tnp_128", "te_tnp_vanilla"]],
        gp_main,
        overrides,
        "AISTATS BSA-TNP - Gaussian Processes",
        dry_run=dry_run,
    )
    # ablation requested by reviewer
    for num_basis_override in [
        "model.blk.attn.attn.bias.s.num_basis=1",
        "model.blk.attn.attn.bias.s.num_basis=3",
        "model.blk.attn.attn.bias.s.num_basis=5",
        "model.blk.attn.attn.bias.s.num_basis=10",
    ]:
        gp_benchmark(
            seeds,
            "2d",
            gp_kernels_2d,
            ["2d/bsa_tnp"],
            gp_main,
            overrides + [num_basis_override],
            "AISTATS BSA-TNP - Gaussian Processes Ablation",
            dry_run=dry_run,
        )
    gp_benchmark(
        seeds,
        "2d_shifted_10",
        gp_kernels_2d,
        [f"2d/{m}" for m in ["bsa_tnp", "tnp_d", "te_tnp", "convcnp_shifted_10"]],
        gp_main,
        overrides + ["project_suffix=' - Shifted 10'", "evaluate_only=True"],
        "AISTATS BSA-TNP - Gaussian Processes",
        dry_run=dry_run,
    )
    gp_benchmark(
        seeds,
        "2d_scaled_2x",
        gp_kernels_2d,
        [f"2d/{m}" for m in ["bsa_tnp", "tnp_d", "te_tnp", "convcnp_scaled_2x"]],
        gp_main,
        overrides + ["project_suffix=' - Scaled 2x'", "evaluate_only=True"],
        "AISTATS BSA-TNP - Gaussian Processes",
        dry_run=dry_run,
    )


def gp_multi_scale_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    generic_benchmark(
        seeds,
        "configs/multiscale_2d_gp",
        ["bsa_tnp", "te_tnp"],
        multiscale_2d_gp_main,
        overrides,
        "AISTATS BSA-TNP - Multiscale Gaussian Processes",
        dry_run=dry_run,
    )


def gp_rotational_invariance_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    rot_models = ["2d/bsa_tnp", "2d/geo_bsa_tnp", "2d/sa_tnp"]
    rot_kernels = ["geo"]
    rots = [  # north, east, tilt
        "60, 30, 0",
        "60, 30, 20",
    ]
    gp_benchmark(
        seeds,
        "so3",
        rot_kernels,
        rot_models,
        gp_main,
        overrides,
        "AISTATS BSA-TNP - Gaussian Processes - Rotated",
        dry_run=dry_run,
    )
    for rot in rots:
        gp_benchmark(
            seeds,
            "so3",
            rot_kernels,
            rot_models,
            gp_main,
            overrides
            + [
                f"project_suffix=' - {rot}'",
                "evaluate_only=True",
                f"data.rotate=[{rot}]",
            ],
            "AISTATS BSA-TNP - Gaussian Processes - Rotated",
            dry_run=dry_run,
        )


def sir_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    models = [
        "bsa_tnp",
        "tnp_d",
        "te_tnp",
        "convcnp",
    ]
    generic_benchmark(
        seeds,
        "configs/sir",
        models,
        sir_main,
        overrides,
        "AISTATS BSA-TNP - SIR",
        dry_run=dry_run,
    )
    # ablation requested by reviewer
    for num_basis_override in [
        "model.blk.attn.attn.bias.s.num_basis=1",
        "model.blk.attn.attn.bias.s.num_basis=3",
        "model.blk.attn.attn.bias.s.num_basis=5",
        "model.blk.attn.attn.bias.s.num_basis=10",
    ]:
        generic_benchmark(
            seeds,
            "configs/sir",
            ["bsa_tnp"],
            sir_main,
            overrides + [num_basis_override],
            "AISTATS BSA-TNP - SIR Ablation",
            dry_run=dry_run,
        )
    generic_benchmark(
        seeds,
        "configs/sir",
        ["bsa_tnp", "tnp_d", "te_tnp", "convcnp_shifted_10"],
        sir_main,
        overrides
        + [
            "project_suffix=' - Shifted 10'",
            "evaluate_only=True",
            "data=spatial_64x64_shifted_10",
        ],
        "AISTATS BSA-TNP - SIR",
        dry_run=dry_run,
    )
    generic_benchmark(
        seeds,
        "configs/sir",
        ["bsa_tnp", "tnp_d", "te_tnp", "convcnp_scaled_2x"],
        sir_main,
        overrides
        + [
            "project_suffix=' - Scaled 2x'",
            "evaluate_only=True",
            "valid_num_steps=1000",
            "data=spatial_128x128",
        ],
        "AISTATS BSA-TNP - SIR",
        dry_run=dry_run,
    )


def era5_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    era5_models = ["tnp_d", "te_tnp", "bsa_tnp"]
    era5_overrides = [
        "data.splits.valid_region=northern_europe",
        "data.splits.test_region=western_europe",
    ]
    generic_benchmark(
        seeds,
        "configs/era5",
        era5_models,
        era5_main,
        overrides + era5_overrides,
        "AISTATS BSA-TNP - ERA5 - CNW",
        dry_run=dry_run,
    )
    # ablation reqeusted by reviewer
    for num_basis_overrides in [
        [
            "model.blk.attn.attn.bias.s.num_basis=1",
            "model.blk.attn.attn.bias.t.num_basis=1",
        ],
        [
            "model.blk.attn.attn.bias.s.num_basis=3",
            "model.blk.attn.attn.bias.t.num_basis=3",
        ],
        [
            "model.blk.attn.attn.bias.s.num_basis=5",
            "model.blk.attn.attn.bias.t.num_basis=5",
        ],
        [
            "model.blk.attn.attn.bias.s.num_basis=10",
            "model.blk.attn.attn.bias.t.num_basis=10",
        ],
    ]:
        generic_benchmark(
            seeds,
            "configs/era5",
            ["bsa_tnp"],
            era5_main,
            overrides + era5_overrides + num_basis_overrides,
            "AISTATS BSA-TNP - ERA5 - CNW Ablation",
            dry_run=dry_run,
        )
    era5_overrides = [
        "data.splits.valid_region=western_europe",
        "data.splits.test_region=northern_europe",
    ]
    generic_benchmark(
        seeds,
        "configs/era5",
        era5_models,
        era5_main,
        overrides + era5_overrides,
        "AISTATS BSA-TNP - ERA5 - CWN",
        dry_run=dry_run,
    )


def beijing_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    beijing_models = [
        "te_tnp",
        "tnp_d",
        "bsa_tnp",
        "bsa_tnp_only_embed",
        "bsa_tnp_only_bias",
    ]
    generic_benchmark(
        seeds,
        "configs/beijing_air_quality",
        beijing_models,
        beijing_air_quality_main,
        overrides,
        "AISTATS BSA-TNP - Beijing Air Quality",
        dry_run=dry_run,
    )


def gneiting_gp_benchmarks(
    seeds: jax.Array,
    overrides: list[str],
    dry_run: bool,
):
    gneiting_gp_models = [
        "bsa_tnp",
        "bsa_tnp_only_embed",
        "bsa_tnp_only_bias",
        "tnp_d",
        "te_tnp",
    ]
    generic_benchmark(
        seeds,
        "configs/gneiting_gp",
        gneiting_gp_models,
        gneiting_gp_main,
        overrides,
        "AISTATS BSA-TNP - Gneiting GP",
        dry_run=dry_run,
    )


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
    main_fn: Callable = gp_main,
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
    bsa_tnp_paper(seeds, args.dry_run)
