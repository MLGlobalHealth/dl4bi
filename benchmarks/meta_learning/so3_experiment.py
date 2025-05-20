import sys

import jax
from gp import main as gp_main
from jax import random

from benchmarks.meta_learning.reproduce_paper import gp_benchmark, parse_args


def so3_experiment(seeds: jax.Array, dry_run: bool = False):
    """SO3 invariance experiments"""
    overrides = []
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides = [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]
    models = [
        "bsa_tnp",
        "geo_bsa_tnp",
        "sa_tnp",
    ]
    models = [f"2d/{m}" for m in models]
    kernels = ["geo"]
    project = "Gaussian Processes - SO3"
    # north, east, tilt
    rotations = [
        "",
        "60, 30, 0",
        "60, 30, 20",
    ]

    # Train
    gp_benchmark(
        seeds,
        "so3",
        kernels,
        models,
        gp_main,
        overrides,
        project,
        dry_run=dry_run,
    )
    # Evaluate SO3 invariance
    for rotation in rotations:
        gp_benchmark(
            seeds,
            "so3",
            kernels,
            models,
            gp_main,
            [
                f"project_suffix=' - Rotated {rotation}'",
                "evaluate_only=True",
                f"data.rotate=[{rotation}]",
            ],
            project,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    args = parse_args(sys.argv)
    rng = random.key(args.seed)
    seeds = random.choice(rng, 100, (args.num_runs,), replace=False)
    so3_experiment(seeds, args.dry_run)
