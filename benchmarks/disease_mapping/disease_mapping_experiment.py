import sys

import jax
from hydra import compose, initialize
from jax import random
from train import main

from benchmarks.meta_learning.reproduce_paper import parse_args


def disease_mapping_experiment(seeds: jax.Array, dry_run: bool = False):
    overrides = [
        "model=spatial/geo_bsa_tnp",
        "data=generic",
        "numpyro=survey_model",
        "input_format=survey",
    ]
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides += [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]

    paths = []
    for seed in seeds:
        for output_format in ["z", "theta"]:
            with initialize(config_path="configs", version_base=None):
                name = f"geotnp-{output_format}"
                cfg = compose(
                    "training",
                    overrides=[
                        "project=Disease Mapping - without covariates",
                        f"seed={seed}",
                        f"output_format={output_format}",
                        f"+name={name}",
                    ]
                    + overrides,
                )
                path = f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{name}"
                paths.append(path)
                print("=" * 100)
                main(cfg)

    for seed in seeds:
        for output_format in ["z", "theta"]:
            with initialize(config_path="configs", version_base=None):
                name = f"geotnp-{output_format}"
                cfg = compose(
                    "training",
                    overrides=[
                        "project=Disease Mapping - with covariates",
                        f"seed={seed}",
                        f"output_format={output_format}",
                        f"+name={name}",
                        "data.urban_rural=true",
                    ]
                    + overrides,
                )
                path = f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{name}"
                paths.append(path)
                print("=" * 100)
                main(cfg)


if __name__ == "__main__":
    args = parse_args(sys.argv)
    rng = random.key(args.seed)
    seeds = random.choice(rng, 100, (args.num_runs,), replace=False)
    disease_mapping_experiment(seeds, args.dry_run)
