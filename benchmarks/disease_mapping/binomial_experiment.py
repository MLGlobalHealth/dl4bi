import sys

import jax
from hydra import compose, initialize
from jax import random
from train import main

from benchmarks.meta_learning.reproduce_paper import parse_args


def binomial_experiment(seeds: jax.Array, dry_run: bool = False):
    overrides = [
        "model=spatial/bsa_tnp",
        "data=1d",
        "numpyro=binomial_model",
        "+model.output_fn={_target_:dl4bi.core.model_output.DiagonalMVNOutput.from_activations, _partial_:true, min_std:0.01 }",
    ]
    if dry_run:
        seeds = seeds[:2]  # no need for more than 2 runs each in dry run
        overrides += [
            "wandb=False",
            "train_num_steps=100",
            "valid_num_steps=50",
            "plot_interval=50",
        ]
    project = "Disease Mapping - Binomial Observations"

    paths = []
    for seed in seeds:
        for input_format in ["survey", "theta", "theta_n", "z", "z_n"]:
            for output_format in ["z", "theta"]:
                with initialize(config_path="configs", version_base=None):
                    name = f"binomial-{input_format}-{output_format}"
                    cfg = compose(
                        "training",
                        overrides=[
                            f"project={project}",
                            f"seed={seed}",
                            f"input_format={input_format}",
                            f"output_format={output_format}",
                            f"+name={name}",
                        ]
                        + overrides,
                    )
                    path = f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{name}"
                    paths.append(path)
                    print("=" * 100)
                    main(cfg)

    seed = 5  # for comparison with mcmc


if __name__ == "__main__":
    args = parse_args(sys.argv)
    rng = random.key(args.seed)
    seeds = random.choice(rng, 100, (args.num_runs,), replace=False)
    binomial_experiment(seeds, args.dry_run)
