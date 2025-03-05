import sys

sys.path.append("benchmarks/vae")
import argparse
import pickle
from pathlib import Path

import arviz as az
import geopandas as gpd
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from infer import load_ckpt
from train import build_spatial_dataloaders
from utils.map_utils import gen_locations
from utils.obj_utils import generate_model_name, instantiate
from utils.plot_utils import (
    plot_infer_trace,
    plot_map_predictive,
    plot_models_mean_prevalence,
    plot_on_map,
    plot_prevalence_scatter_comp,
    plot_vae_reconstruction,
    plot_vae_scatter_plot,
)

from dl4bi.vae.train_utils import TrainState


def reproduce_plots(seeds: jax.Array):
    models_cfgs = ["auto_deep_RV", "deep_RV_gMLP", "auto_prior_cvae"]
    models_n = ["DeepRV_MLP", "DeepRV_gMLP", "PriorCVAE_MLP", "Baseline_GP"]
    spatial_priors = ["rbf", "matern_3_2", "matern_1_2", "matern_5_2", "car"]
    seed = int(seeds[0])
    plot_vae_train_samples(seed, models_cfgs, spatial_priors)
    plot_empirical_bayes_comparison(seed, models_n, spatial_priors)
    infer_summary = summarize_inference_runs(
        seed,
        models_n,
        spatial_priors,
        "UK_LTLA_sim",
        "benchmarks/vae/maps/UK",
        "poisson",
    )
    print_simulated_latex_table(pd.DataFrame(infer_summary), models_n)
    summarize_inference_runs(
        seed,
        models_n,
        ["matern_1_2", "matern_3_2"],
        "female_U50_cancer_mort",
        "benchmarks/vae/maps/female_under_50_cancer_mortality_LAD_2023",
        "binomial",
    )
    summarize_inference_runs(
        seed,
        models_n,
        ["matern_1_2", "matern_3_2"],
        "male_U50_cancer_mort",
        "benchmarks/vae/maps/male_under_50_cancer_mortality_LAD_2023",
        "binomial",
    )
    summarize_inference_runs(
        seed,
        models_n,
        ["matern_1_2"],
        "zimbabwe_HIV",
        "benchmarks/vae/maps/zwe2016phia_fixed.geojson",
        "binomial",
        population_scale=1,
    )


def plot_vae_train_samples(seed: int, models: list[str], spatial_priors: list[str]):
    exp_name = "UK_LTLA_sim"
    map_path = "benchmarks/vae/maps/UK"
    save_dir = Path("results/final_plots/vae/")

    for spatial_prior_name in spatial_priors:
        for model in models:
            with hydra.initialize(config_path="../configs", version_base=None):
                cfg = hydra.compose(
                    "default",
                    overrides=[
                        f"exp_name={exp_name}",
                        f"data.map_path={map_path}",
                        f"inference_model.spatial_prior.func={spatial_prior_name}",
                        f"seed={seed}",
                        f"model={model}",
                    ],
                )
                model_name = generate_model_name(cfg)
                model_save_dir = save_dir / f"{spatial_prior_name}/{model_name}/"
                model_save_dir.mkdir(parents=True, exist_ok=True)
                spatial_prior = instantiate(cfg.inference_model.spatial_prior)
                model_dir = Path(
                    f"results/{cfg.exp_name}/{spatial_prior_name}/{cfg.seed}"
                )
                map_data, s = gen_locations(cfg.data)
                kwargs = {}
                if "FixedLocationTransfomer" in model_name:
                    kwargs = {"s": s}
                state, _ = load_ckpt((model_dir / model_name).with_suffix(".ckpt"))
                priors = {
                    pr: instantiate(pr_dist)
                    for pr, pr_dist in cfg.inference_model.priors.items()
                }
                rng = jax.random.key(seed)
                loader, _, _, cond_names = build_spatial_dataloaders(
                    rng, cfg, map_data, s, priors, spatial_prior
                )
                plot_vae_reconstruction(
                    rng,
                    s,
                    map_data,
                    state,
                    model_name,
                    loader,
                    cond_names,
                    "DeepRV" in model_name,
                    save_dir=model_save_dir,
                    **kwargs,
                )
                plot_vae_scatter_comp(
                    rng,
                    state,
                    model_name,
                    loader,
                    cond_names,
                    "DeepRV" in model_name,
                    model_save_dir,
                    **kwargs,
                )


def plot_vae_scatter_comp(
    rng: jax.Array,
    state: TrainState,
    model: str,
    loader,
    conds_names: list[str],
    is_decoder_only: bool,
    save_dir: Path,
    num_samples=5,
    **kwargs,
):
    rng_drop, rng_extra, rng = jax.random.split(rng, 3)
    f, z, conditionals = next(loader)
    f_hat = state.apply_fn(
        {"params": state.params, **state.kwargs},
        z if model == "DeepRV" else f,
        conditionals,
        **kwargs,
        rngs={"dropout": rng_drop, "extra": rng_extra},
    )
    f_hat = f_hat if is_decoder_only else f_hat[0]

    plot_vae_scatter_plot(
        f, f_hat, None, conds_names, num_samples=num_samples, save_dir=save_dir
    )


def plot_empirical_bayes_comparison(
    seed: int,
    models: list[str],
    spatial_priors: list[str],
):
    exp_name = "UK_LTLA_sim"
    save_dir = Path("results/final_plots/EB/")
    abs_min, abs_max = 10000, -10000
    for spatial_prior in spatial_priors:
        compare_var = "alpha" if spatial_prior == "car" else "ls"
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
        axes = axes.flatten() if len(models) > 1 else [axes]
        for model, ax in zip(models, axes):
            result_dir = Path(
                f"results/{exp_name}/{spatial_prior}/{seed}/{model}/Empirical_Bayes"
            )
            with open(result_dir / "cond_data.pkl", "rb") as ff:
                data = pickle.load(ff)
                inf_c = data[f"inf_{compare_var}"]
                real_c = data[f"real_{compare_var}"]
            ax.scatter(real_c, inf_c, alpha=0.6)
            ax.set_title(model.replace("_", " "))
            abs_min = min(abs_min, min(real_c.min(), inf_c.min()))
            abs_max = max(abs_max, max(real_c.max(), inf_c.max()))
        for ax in fig.axes:
            ax.plot([abs_min, abs_max], [abs_min, abs_max], "r--", label="y = x")
            ax.set_xlabel(f"Real {compare_var.replace('ls', 'lengthscale')}")
            ax.set_ylabel(f"EB Estimate of {compare_var.replace('ls', 'lengthscale')}")
            ax.legend(loc="lower right")
            ax.set_xlim(abs_min - 0.01, abs_max + 0.01)
            ax.set_ylim(abs_min - 0.01, abs_max + 0.01)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_path = str(save_dir / f"Scatter_{spatial_prior}.png")
        fig.subplots_adjust(top=0.86)
        fig.savefig(plot_path, dpi=125)
        plt.clf()
        plt.close(fig)


def summarize_inference_runs(
    seed: int,
    models: list[str],
    spatial_priors: list[str],
    exp_name: str,
    map_path: str,
    model_type: str,
    population_scale: int = 100,
    log_plot: bool = True,
):
    rng, _ = jax.random.split(jax.random.key(seed))
    save_dir = Path(
        f"results/final_plots/{'simulated_data' if exp_name == 'UK_LTLA_sim' else 'real_data'}"
    )
    map_data = gpd.read_file(map_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    infer_summary = []
    for spatial_prior in spatial_priors:
        f_hats = [None]
        prev_hats = []
        prev_real = None
        for i, model in enumerate(models):
            result_dir = Path(
                f"results/{exp_name}/{spatial_prior}/{seed}/{model}/{model_type}/complete_info/"
            )
            with open(result_dir / "hmc_pp.pkl", "rb") as ff:
                post = pickle.load(ff)
            with open(result_dir / "hmc_samples.pkl", "rb") as ff:
                samples = pickle.load(ff)
            with open(result_dir / "mcmc.pkl", "rb") as ff:
                mcmc = pickle.load(ff)
            inferred_var = "alpha" if spatial_prior == "car" else "ls"
            mcmc_sum = az.summary(mcmc, round_to=3)
            ess = az.ess(mcmc, method="mean")
            lambda_hat = samples["beta"][..., None] + samples["mu"]
            prev_hats.append(lambda_hat)
            f, f_hat = post["f"], post["obs"]
            if (
                i == 0
                and post.get("beta") is not None
                and post.get("spatial_eff") is not None
            ):
                prev_real = post["beta"] + post["spatial_eff"]
            f_hats[0] = f
            f_hats.append(f_hat)
            infer_summary.append(
                {
                    "model": model,
                    "spatial_prior": spatial_prior,
                    "Inferred var": mcmc_sum.at[inferred_var, "mean"].item(),
                    "r_hat": mcmc_sum.at[inferred_var, "r_hat"].item(),
                    "ESS Inferred var": ess[inferred_var].item(),
                    "Mean ESS GP": ess["mu"].mean().item(),
                    "MSE(f, f_pred)": ((f - f_hat.mean(axis=0)) ** 2).mean(),
                }
            )
            plot_infer_obs_summary(
                f,
                f_hat,
                map_data,
                save_dir / f"{exp_name}_{model}_{spatial_prior}_infer_summary.png",
                log=log_plot,
            )
            plot_infer_trace(
                samples,
                mcmc,
                conditionals=None,
                var_names=["alpha" if spatial_prior == "car" else "ls"]
                + (["var", "beta"] if model_type == "binomial" else []),
                save_path=save_dir
                / f"{exp_name}_{model}_{spatial_prior}_infer_trace.png",
            )
            plot_map_predictive(
                rng,
                f,
                f_hat,
                map_data,
                save_dir / f"{exp_name}_{model}_{spatial_prior}_infer_samples.png",
                log=log_plot,
            )
        plot_models_predictive_means(
            f_hats,
            map_data,
            models,
            save_dir / f"{exp_name}_{spatial_prior}_predictive_means.png",
            log=log_plot,
        )
        population = (
            None
            if "population" not in map_data.columns
            else jnp.array(map_data.population)
        )
        plot_models_mean_prevalence(
            prev_real,
            prev_hats,
            f,
            population,
            models,
            model_type,
            map_data,
            save_dir / f"{exp_name}_{spatial_prior}_prevalence_means.png",
            population_scale=population_scale,
        )
        plot_prevalence_scatter_comp(
            prev_real,
            prev_hats,
            f,
            population,
            models,
            model_type,
            save_dir / f"{exp_name}_{spatial_prior}_prevalence_scatter.png",
            population_scale=population_scale,
        )
    return infer_summary


def plot_models_predictive_means(
    f_hats,
    map_data,
    models,
    save_path: Path,
    log=True,
):
    if log:
        f_hats = [jnp.log(f_mean + 1) for f_mean in f_hats]
    f_hat_means = [f_hats[0]] + [f_mean.mean(axis=0) for f_mean in f_hats[1:]]
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    fig, ax = plt.subplots(1, len(f_hat_means), figsize=(6 * len(f_hat_means), 8))
    log_str = " (Log scale)" if log else ""
    for i, f_mean in enumerate(f_hat_means):
        title = "Observed counts" if i == 0 else f"{models[i - 1]}: Mean estimate"
        plot_on_map(ax[i], map_data, f_mean, vmin, vmax, f"{title}{log_str}")

    for axis in ax:
        axis.set_axis_off()
    plt.tight_layout()
    fig.savefig(save_path, dpi=125)
    plt.clf()
    plt.close(fig)


def plot_models_mean_lambda(
    lambda_hats,
    map_data,
    models,
    save_path: Path,
    log=False,
):
    if log:
        lambda_hats = [jnp.log(f_mean + 1) for f_mean in lambda_hats]
    lambda_hat_means = [f_mean.mean(axis=0) for f_mean in lambda_hats]
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in lambda_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in lambda_hat_means])).item()
    fig, ax = plt.subplots(
        1, len(lambda_hat_means), figsize=(6 * len(lambda_hat_means), 8)
    )
    log_str = " (Log scale)" if log else ""
    for i, f_mean in enumerate(lambda_hat_means):
        title = f"{models[i]}: Mean estimate"
        plot_on_map(ax[i], map_data, f_mean, vmin, vmax, f"{title}{log_str}")
    for axis in ax:
        axis.set_axis_off()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.clf()
    plt.close(fig)


def plot_infer_obs_summary(f, f_hat, map_data, save_path: Path, log=True):
    if log:
        f, f_hat = jnp.log(f + 1), jnp.log(f_hat + 1)
    f_hat_mean, f_hat_std = f_hat.mean(axis=0), f_hat.std(axis=0)
    vmin, vmax = min(f.min(), f_hat_mean.min()), max(f.max(), f_hat_mean.max())
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    log_str = " (Log scale)" if log else ""
    plot_on_map(ax[0], map_data, f, vmin, vmax, f"Observed counts{log_str}")
    plot_on_map(ax[1], map_data, f_hat_mean, vmin, vmax, f"Mean estimate{log_str}")
    plot_on_map(
        ax[2], map_data, f_hat_std, title=f"Standard deviation{log_str}", cmap="plasma"
    )
    for axis in ax:
        axis.set_axis_off()
    plt.tight_layout()
    fig.savefig(save_path, dpi=125)
    plt.clf()
    plt.close(fig)


def print_simulated_latex_table(infer_summary, models_n):
    spatial_priors = ["car", "matern_1_2", "matern_3_2", "matern_5_2", "rbf"]
    latex_str = r"""\begin{table}
\small
\centering
\caption{Simulated data: inference results, single run}
\label{table:sim_infer}
\begin{tabular}{l|l l l l l l}
\toprule
\multirow{2}{*} &  & \textbf{CAR} & \textbf{Mat\'ern-1/2} & \textbf{Mat\'ern-3/2} & \textbf{Mat\'ern-5/2} & \textbf{RBF} \\
 & & ($\alpha$ - Eq \ref{eqaution:CAR}) & (length scale) & (length scale) & (length scale) & (length scale) \\
\midrule
\multirow{1}{*}{\begin{sideways}\textbf{}\end{sideways}} 
& Real Value & 0.95 & 0.2 & 0.2 & 0.2 & 0.2 \\
\midrule
"""

    for model in models_n:
        latex_str += (
            rf"\multirow{{6}}{{*}}{{\begin{{sideways}}\textbf{{{model}}}\end{{sideways}}}} "
            + "\n"
        )
        for metric, label, is_int in zip(
            [
                "Inferred var",
                "ESS Inferred var",
                "r_hat",
                "MSE(f, f_pred)",
                "Mean ESS GP",
                "Runtime",
            ],
            [
                "Inferred Value",
                "ESS Inferred Variable",
                r"$\hat{r}$",
                r"MSE($f$, $\bar{f}_{\text{pred}}$)",
                "Mean ESS GP",
                "Runtime (s)",
            ],
            [False, True, False, False, True, True],  # Which columns should be int
        ):
            row_values = []
            for spatial_prior in spatial_priors:
                if metric == "Runtime":
                    row_values.append("-1")
                    continue
                value = infer_summary.loc[
                    (infer_summary["model"] == model)
                    & (infer_summary["spatial_prior"] == spatial_prior),
                    metric,
                ]
                if not value.empty:
                    if is_int:
                        row_values.append(f"{int(value.values[0])}")
                    else:
                        row_values.append(f"{value.values[0]:.3f}")
                else:
                    row_values.append("-")  # Placeholder if no value exists
            latex_str += f" & {label} & " + " & ".join(row_values) + r" \\" + "\n"
        latex_str += r"\midrule" + "\n"
    latex_str += r"\bottomrule" + "\n\\end{tabular}\n\\end{table}"
    print(latex_str)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # NOTE: to exactly get the same seed as deep_rv_experiments.py
    seeds = jax.random.choice(
        jax.random.key(args.seed),
        jax.numpy.arange(100),
        shape=(args.num_runs,),
        replace=False,
    )
    reproduce_plots(seeds)
