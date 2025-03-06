import sys

sys.path.append("benchmarks/vae")
import argparse

import hydra
import jax
from empirical_bayes import main as infer_empirical_bayes_main
from infer import main as infer_main
from train import main as vae_main


def run_experiments(
    seeds: jax.Array,
    simulated_data: bool,
    real_data: bool,
    dry_run: bool,
):
    """Run the full experiment for all configurations."""
    vae_overrides = []
    infer_overrides = []
    if dry_run:
        seeds = seeds[:1]
        vae_overrides += [
            "plot_interval=100000",
            "valid_interval=100",
            "train_num_steps=100",
            "valid_num_steps=100",
            "wandb=False",
        ]
        infer_overrides += [
            "mcmc.num_samples=50",
            "mcmc.num_warmup=10",
            "mcmc.num_chains=1",
            "wandb=False",
        ]
    if simulated_data:
        exp_name = "1d_gp"
        spatial_priors = ["matern_5_2"]
        models = ["auto_prior_cvae"]
        run_vae_train(
            "priorcvae_v4",
            exp_name,
            seeds,
            spatial_priors,
            models,
            vae_overrides,
        )


def run_vae_train(
    project: str,
    exp_name: str,
    seeds: jax.Array,
    spatial_priors: list[str],
    models: list[str],
    vae_overrides: list[str],
):
    for seed in seeds:
        for spatial_prior in spatial_priors:
            for model in models:
                with hydra.initialize(config_path="../configs", version_base=None):
                    cfg = hydra.compose(
                        "default",
                        overrides=vae_overrides
                        + [
                            f"project={project}",
                            f"exp_name={exp_name}",
                            f"inference_model=priorcvae_gp_train",
                            f"inference_model.spatial_prior.func={spatial_prior}",
                            f"seed={seed}",
                            f"model={model}",
                            f"data=1d"
                        ],
                    )
                    print("Running vae.py")
                    vae_main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=17,
        help="Initial seed",
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
    parser.add_argument(
        "--simulated_data",
        action="store_true",
        default=False,
        help="Whether to run the simulated data experiments",
    )
    parser.add_argument(
        "--real_data",
        action="store_true",
        default=False,
        help="Whether to run the real data experiments",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = jax.random.choice(
        jax.random.key(args.seed),
        jax.numpy.arange(100),
        shape=(args.num_runs,),
        replace=False,
    )
    run_experiments(seeds, args.simulated_data, args.real_data, args.dry_run)
