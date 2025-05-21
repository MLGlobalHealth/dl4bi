import sys

import jax
from hydra import compose, initialize
from jax import random
from train import main

from benchmarks.meta_learning.reproduce_paper import parse_args


def binomial_experiment(seeds: jax.Array, dry_run: bool = False):
    overrides = []
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides = [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]
    project = "Disease Mapping - Binomial Observations"

    for seed in seeds:
        for input_format in ["survey", "theta", "theta_n", "z", "z_n"]:
            for output_format in ["z", "theta"]:
                with initialize(config_path="configs", version_base=None):
                    cfg = compose(
                        "training",
                        overrides=[
                            f"project={project}",
                            "model=spatial/bsa_tnp",
                            "data=1d",
                            "numpyro=binomial_model",
                            f"seed={seed}",
                            f"input_format={input_format}",
                            f"output_format={output_format}",
                            f"+name=binomial-{input_format}-{output_format}",
                        ]
                        + overrides,
                    )
                    print("=" * 100)
                    main(cfg)


if __name__ == "__main__":
    args = parse_args(sys.argv)
    rng = random.key(args.seed)
    seeds = random.choice(rng, 100, (args.num_runs,), replace=False)
    binomial_experiment(seeds, args.dry_run)
