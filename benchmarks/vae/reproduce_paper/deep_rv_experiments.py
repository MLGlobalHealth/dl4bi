import argparse

import hydra
import jax
from infer import main as infer_main
from model_evaluations.infer_empirical_bayes import main as infer_empirical_bayes_main
from vae import main as vae_main


def run_experiments(
    seeds: jax.Array,
    deep_rv_comp: bool,
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

    if deep_rv_comp:
        exp_name = "UK_LTLA_sim"
        map_path = "maps/UK"
        spatial_priors = ["rbf", "matern_3_2", "matern_1_2", "matern_5_2", "car"]
        models = ["deep_RV_UK_sim", "deep_RV_gMLP", "deep_RV_RBF_net"]
        run_vae_train(
            "deep_RV_sim",
            map_path,
            exp_name,
            seeds[:2],
            spatial_priors,
            models,
            vae_overrides,
        )
        # NOTE: only runs on one seed, comparison to full MCMC is required which
        # takes too long for multiple seeds
        run_inference(
            "deep_RV_sim",
            map_path,
            exp_name,
            seeds[:1],
            spatial_priors,
            models + ["Baseline_GP"],
            infer_overrides,
        )

    if simulated_data:
        exp_name = "UK_LTLA_sim"
        map_path = "maps/UK"
        spatial_priors = ["rbf", "matern_3_2", "matern_1_2", "matern_5_2", "car"]
        models = ["deep_RV", "prior_cvae"]
        run_vae_train(
            "deep_RV_sim",
            map_path,
            exp_name,
            seeds,
            spatial_priors,
            models,
            vae_overrides,
        )
        run_empirical_bayes(
            "deep_RV_sim",
            map_path,
            exp_name,
            seeds,
            spatial_priors,
            models + ["Baseline_GP"],
            infer_overrides,
        )
        # NOTE: only runs on one seed, comparison to full MCMC is required which
        # takes too long for multiple seeds
        run_inference(
            "deep_RV_sim",
            map_path,
            exp_name,
            seeds[:1],
            spatial_priors,
            models + ["Baseline_GP"],
            infer_overrides,
        )

    if real_data:
        exp_names = ["total_U50_cancer_mort", "zimbabwe_HIV"]
        map_paths = [
            "maps/total_under_50_cancer_mortality_LAD_2023",
            "maps/zwe2016phia_fixed.geojson",
        ]
        model_types = ["binomial_real"]
        spatial_prior_per_exp = [["matern_3_2"], ["matern_1_2"]]
        models = ["deep_RV"]
        for exp_name, map_path, spatial_priors in zip(
            exp_names, map_paths, spatial_prior_per_exp
        ):
            pop_scale = 100 if exp_name == "total_U50_cancer_mort" else 1
            run_vae_train(
                f"deep_RV_{exp_name}",
                map_path,
                exp_name,
                seeds[:1],
                spatial_priors,
                models,
                vae_overrides,
            )
            for model_type in model_types:
                run_inference(
                    f"deep_RV_{exp_name}",
                    map_path,
                    exp_name,
                    seeds[:1],
                    spatial_priors,
                    models + ["Baseline_GP"],
                    infer_overrides + [f"inference_model.population_scale={pop_scale}"],
                    model_type=model_type,
                )


def run_vae_train(
    project: str,
    map_path: str,
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
                        "default" + ("_car" if spatial_prior == "car" else ""),
                        overrides=vae_overrides
                        + [
                            f"project={project}",
                            f"exp_name={exp_name}",
                            f"map_path={map_path}",
                            f"inference_model.spatial_prior.func={spatial_prior}",
                            f"seed={seed}",
                            f"model={model}",
                        ],
                    )
                    print("Running vae.py")
                    vae_main(cfg)


def run_empirical_bayes(
    project: str,
    map_path: str,
    exp_name: str,
    seeds: jax.Array,
    spatial_priors: list[str],
    models: list[str],
    infer_overrides: list[str],
):
    for seed in seeds:
        for spatial_prior in spatial_priors:
            for model in models:
                inference_model = "inference_model=spatial_only"
                if spatial_prior == "car":
                    inference_model += "_car"
                with hydra.initialize(config_path="../configs", version_base=None):
                    overrides = infer_overrides + [
                        f"project={project}",
                        f"exp_name={exp_name}",
                        f"map_path={map_path}",
                        "inference_model=spatial_only",
                        f"seed={seed}",
                    ]
                    if model == "Baseline_GP":
                        overrides += ["inference_model.surrogate_model=False"]
                    else:
                        overrides += [f"model={model}"]
                    cfg = hydra.compose(
                        "default" + ("_car" if spatial_prior == "car" else ""),
                        overrides=overrides,
                    )
                    cfg.inference_model.spatial_prior.func = f"{spatial_prior}"
                    print("Running infer_empirical_bayes.py")
                    infer_empirical_bayes_main(cfg)


def run_inference(
    project: str,
    map_path: str,
    exp_name: str,
    seeds: jax.Array,
    spatial_priors: list[str],
    models: list[str],
    infer_overrides: list[str],
    model_type: str = "poisson",
):
    for seed in seeds:
        for spatial_prior in spatial_priors:
            for model in models:
                with hydra.initialize(config_path="../configs", version_base=None):
                    overrides = infer_overrides + [
                        f"project={project}",
                        f"exp_name={exp_name}",
                        f"map_path={map_path}",
                        f"inference_model.spatial_prior.func={spatial_prior}",
                        f"seed={seed}",
                    ]
                    if model == "Baseline_GP":
                        overrides += ["inference_model.surrogate_model=False"]
                    else:
                        overrides += [f"model={model}"]
                    cfg = hydra.compose(
                        "default" + ("_car" if spatial_prior == "car" else ""),
                        overrides=overrides,
                    )
                    print("Running infer.py")
                    infer_main(cfg)


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
        "--deep_rv_comp",
        action="store_true",
        default=False,
        help="Whether to run architecture comparisom for deepRV",
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
    run_experiments(
        seeds, args.deep_rv_comp, args.simulated_data, args.real_data, args.dry_run
    )
