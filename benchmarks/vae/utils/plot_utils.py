from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import arviz as az
import geopandas as gpd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import TrainState
from utils.map_utils import get_norm_vars


def _to_prev(prev_hat: jax.Array, inference_model: str):
    if inference_model == "poisson":
        return jnp.exp(prev_hat)
    elif inference_model == "binomial":
        return 1 / (1 + jnp.exp(-prev_hat))
    else:
        return prev_hat


def plot_prevalence_scatter_comp(
    prev_real: Optional[jax.Array],
    prev_hats: list[jax.Array],
    f_obs: jax.Array,
    population: Optional[jax.Array],
    models: list[str],
    inference_model: str,
    save_path: Optional[Union[Path, str]] = None,
    population_scale: int = 100,
    log: bool = False,
):
    if prev_real is not None:
        prev_real = _to_prev(prev_real, inference_model)
    elif inference_model == "binomial" and population is not None:
        population = population // population_scale
        prev_real = jnp.array(f_obs / population)
    if prev_real is None:
        return
    val_n = "Prevalence" if inference_model == "binomial" else "Intensity"
    prev_hats = [_to_prev(prev_hat, inference_model) for prev_hat in prev_hats]
    if log:
        prev_hats = [jnp.log(prev_hat + 1) for prev_hat in prev_hats]
        prev_real = jnp.log(prev_real + 1)
    prev_hat_means = [prev_hat.mean(axis=0) for prev_hat in prev_hats]
    abs_min, abs_max = prev_real.min(), prev_real.max()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 8))
    axes = [axes] if len(models) == 1 else axes
    for i, (ax, model) in enumerate(zip(axes, models)):
        p_hat_i = prev_hat_means[i]
        ax.scatter(prev_real, p_hat_i, alpha=0.6, label="Samples")
        model_n = model.replace("Baseline_", "").replace("_", " + ")
        ax.set_title(f"{val_n} vs. {model_n} mean {val_n.lower()}")
        abs_min = min(abs_min, p_hat_i.min())
        abs_max = max(abs_max, p_hat_i.max())
    for ax in axes:
        ax.plot([abs_min, abs_max], [abs_min, abs_max], "r--", label="y = x")
        ax.set_xlabel(r"$p$" if inference_model == "binomial" else r"$\lambda$")
        ax.set_ylabel(
            r"$\hat{p}$" if inference_model == "binomial" else r"$\hat{\lambda}$"
        )
        ax.legend(loc="lower right")
        ax.set_xlim(abs_min - 0.01, abs_max + 0.01)
        ax.set_ylim(abs_min - 0.01, abs_max + 0.01)
    plt.tight_layout()
    fig.subplots_adjust(top=0.86)
    if save_path is None:
        save_path = f"/tmp/Scatter prevalence{datetime.now().isoformat()}.png"
    fig.savefig(save_path, dpi=125)
    plt.clf()
    plt.close(fig)


def plot_models_mean_prevalence(
    prev_real: Optional[jax.Array],
    prev_hats: list[jax.Array],
    f_obs: jax.Array,
    population: Optional[jax.Array],
    models: list[str],
    inference_model: str,
    map_data: gpd.GeoDataFrame,
    save_path: Optional[Union[Path, str]] = None,
    population_scale: int = 100,
    log: bool = False,
):
    if prev_real is not None:
        prev_real = _to_prev(prev_real, inference_model)
    elif inference_model == "binomial" and population is not None:
        population = population // population_scale
        prev_real = jnp.array(f_obs / population)
    val_n = "Prevalence" if inference_model == "binomial" else "Intensity"
    prev_hats = [_to_prev(prev_hat, inference_model) for prev_hat in prev_hats]
    use_real = prev_real is not None
    if log:
        prev_hats = [jnp.log(prev_hat + 1) for prev_hat in prev_hats]
        prev_real = jnp.log(prev_real + 1) if use_real else prev_real
    prev_hat_means = [prev_hat.mean(axis=0) for prev_hat in prev_hats]
    if use_real:
        prev_hat_means = [prev_real] + prev_hat_means
    vmin = jnp.min(jnp.array([prev_mean.min() for prev_mean in prev_hat_means])).item()
    vmax = jnp.max(jnp.array([prev_mean.max() for prev_mean in prev_hat_means])).item()
    fig, axes = plt.subplots(
        1,
        len(prev_hat_means),
        figsize=(6 * len(prev_hat_means), 7),
        constrained_layout=True,
    )
    log_str = " (Log scale)" if log else ""
    for i, prev_mean in enumerate(prev_hat_means):
        model_n = models[i - 1].replace("Baseline_", "").replace("_", " + ")
        legend = i == len(prev_hat_means) - 1
        title = f"{model_n}: Mean {val_n.lower()}"
        if i == 0 and use_real:
            title = f"Observed {val_n.lower()}{log_str}"
        if save_path is not None:
            title = ""
        ax = axes if len(prev_hat_means) == 1 else axes[i]
        plot_on_map(ax, map_data, prev_mean, vmin, vmax, title, legend=legend)
        ax.set_axis_off()
        ax.set_title(ax.get_title(), fontsize=16)
    if save_path is None:
        save_path = f"/tmp/Observed prevalence{datetime.now().isoformat()}.png"
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


def plot_map_predictive(
    rng: jax.Array,
    f: jax.Array,
    f_hat: jax.Array,
    map_data: gpd.GeoDataFrame,
    save_path: Optional[Union[Path, str]] = None,
    log: bool = True,
    n_samples: int = 3,
):
    log_str = " (Log scale)" if log else ""
    if log:
        f, f_hat = jnp.log(f + 1), jnp.log(f_hat + 1)
    idxs = jax.random.choice(
        rng, jnp.arange(f_hat.shape[0]), (n_samples,), replace=False
    )
    fig, ax = plt.subplots(1, n_samples + 1, figsize=((n_samples + 1) * 5, 8))
    plot_on_map(ax[0], map_data, f, title=f"Observed counts{log_str}", legend=False)
    ax[0].set_axis_off()
    for i, s_idx in enumerate(idxs):
        plot_on_map(
            ax[i + 1],
            map_data,
            f_hat[s_idx],
            title=f"Posterior Predictive Sample {i + 1}{log_str}",
            legend=False,
        )
        ax[i + 1].set_axis_off()
    plt.tight_layout()
    if not save_path:
        save_path = f"/tmp/Inferred Realisations {datetime.now().isoformat()}.png"
        fig.savefig(save_path, dpi=300)
        wandb.log({f"Inferred Realisations{log_str}": wandb.Image(save_path)})
    else:
        fig.savefig(save_path, dpi=300)
    fig.clf()
    plt.close(fig)


def plot_infer_trace(
    samples,
    mcmc,
    conditionals: Optional[dict] = None,
    var_names: Optional[list[str]] = None,
    save_path: Optional[Path] = None,
):
    if isinstance(mcmc, numpyro.infer.MCMC):
        mcmc = az.from_numpyro(mcmc)
    if var_names is None and conditionals is not None:
        var_names = [
            str(c)
            for c in conditionals.keys()
            if c in mcmc.posterior.data_vars.variables
        ]
    if len(var_names) == 0:
        return
    if save_path is None:
        save_path = Path(f"/tmp/trace_{datetime.now().isoformat()}.png")
    az.plot_trace(mcmc, var_names=var_names)
    conditional_means = {c: samples[str(c)].mean().item() for c in var_names}
    axes = plt.gcf().get_axes()
    for i, (name, mean_val) in enumerate(conditional_means.items()):
        axes[i * 2].set_title(f"{name} (mean: {mean_val:.2f})", fontsize=10)
        axes[(i * 2) + 1].set_title(f"{name} (mean: {mean_val:.2f})", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.clf()
    plt.close()


def plot_vae_reconstruction(
    rng: jax.Array,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    state: TrainState,
    model: str,
    loader,
    conds_names: list[str],
    save_dir: Optional[Path] = None,
    num_plots: int = 5,
    samples_per_plot: int = 3,
    plot_locations=False,
    step="",
):
    """Plots VAE predictions on map"""
    x_norm_vars, y_norm_vars = get_norm_vars(map_data)
    x_mean, x_std = x_norm_vars
    y_mean, y_std = y_norm_vars
    paths = []
    for i in range(num_plots):
        rng_drop, rng_extra, rng = jax.random.split(rng, 3)
        batch = next(loader)
        f, conditionals = batch["f"], batch["conditionals"]
        output: VAEOutput = state.apply_fn(
            {"params": state.params, **state.kwargs},
            **batch,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        f_hat = output.f_hat
        fig, ax = plt.subplots(
            1, samples_per_plot * 2 + int(plot_locations), figsize=(16, 5)
        )
        for j in range(samples_per_plot):
            f_j = f[j].squeeze()
            f_hat_j = f_hat[j].squeeze()
            vmin = min(f_j.min(), f_hat_j.min())
            vmax = max(f_j.max(), f_hat_j.max())
            plot_on_map(ax[2 * j], map_data, f_j, vmin, vmax, f"Sample {j}: " + r"$f$")
            plot_on_map(
                ax[2 * j + 1],
                map_data,
                f_hat_j,
                vmin,
                vmax,
                f"Sample {j}: " + r"$\hat{f}$",
            )
        if plot_locations:
            context_points = np.array(
                [(s[:, 0] * x_std) + x_mean, (s[:, 1] * y_std) + y_mean]
            ).T
            plot_locations_map(ax[-1], map_data, context_points, plot_blank=True)
        for axis in ax:
            axis.set_axis_off()
        plt.tight_layout()
        title = f"{model}, {conds_to_title(conds_names, conditionals)}"
        fig.suptitle(title)
        fig.subplots_adjust(top=0.85)
        if save_dir:
            fig.savefig(save_dir / f"rec_{i}.png", dpi=125)
        else:
            paths.append(f"/tmp/VAE_Rec {datetime.now().isoformat()} - {title}.png")
            fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    if save_dir is None:
        wandb.log({f"Reconstruction {step}": [wandb.Image(p) for p in paths]})


def plot_vae_scatter_plot(
    f: jax.Array,
    f_hat: jax.Array,
    conditionals: Optional[list],
    conds_names: list[str],
    num_samples: int = 5,
    save_dir: Optional[Path] = None,
    step="",
):
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    for i, ax in enumerate(axes.flatten() if num_samples > 1 else [axes]):
        f_i, f_hat_i = f[i], f_hat[i]
        ax.scatter(f_i, f_hat_i, alpha=0.6, label="Samples")
        min_val = min(f_i.min(), f_hat_i.min())
        max_val = max(f_i.max(), f_hat_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
        ax.set_xlabel(r"$f$")
        ax.set_ylabel(r"$\hat{f}$")
        ax.legend()
    plt.tight_layout()
    fig.subplots_adjust(top=0.86)
    if save_dir:
        fig.savefig(save_dir / "scatter.png", dpi=125)
    else:
        timestamp = datetime.now().isoformat()
        title = "Y vs Y hat"
        if conditionals is not None:
            title = f"{title} {conds_to_title(conds_names, conditionals)}"
        path = f"/tmp/VAE_Scatter {timestamp} - {title}.png"
        fig.savefig(path, dpi=125)
        wandb.log({f"Scatter {step}": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


def plot_on_map(
    ax,
    gdf: gpd.GeoDataFrame,
    values: jax.Array,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    cmap: str = "viridis",
    legend: bool = True,
):
    ax.set_title(title)
    gdf["TEMP"] = values
    gdf.plot(column="TEMP", cmap=cmap, ax=ax, legend=legend, vmin=vmin, vmax=vmax)


def plot_locations_map(
    ax,
    gdf: gpd.GeoDataFrame,
    locations,
    sampling_policy: str = "centroids",
    plot_blank: bool = False,
):
    if plot_blank:
        gdf.plot(ax=ax)
    ax.set_title(f"{len(locations)} {sampling_policy.replace('_', ' ')}")
    ax.scatter(locations[:, 0], locations[:, 1], color="red", marker=".", s=15)


def conds_to_title(conds_names: list[str], conditionals: list[jax.Array]):
    if None in conditionals:
        return ""
    return (
        "("
        + ", ".join(
            [
                f"{conds_names[j]}: {conditionals[j]:0.2f}"
                for j in range(len(conds_names))
            ]
        )
        + ")"
    )


def plot_posterior_predictive_comparisons(
    samples: list[dict],
    conditionals: dict,
    priors: dict,
    model_names: list[str],
    var_names: list[str],
    save_prefix: Path,
    baseline_model: str = "Baseline_GP",
):
    baseline_index = model_names.index(baseline_model)
    for var_name in var_names:
        actual_val = conditionals.get(var_name, None)
        fig, ax = plt.subplots(figsize=(4, 4))
        min_val, max_val = 1000, -1000
        for model_name in model_names:
            model_idx = model_names.index(model_name)
            model_samples = samples[model_idx].get(str(var_name), None)
            if model_samples is not None:
                min_val = min(min_val, model_samples.min())
                max_val = max(max_val, model_samples.max())
                model_n = model_name.replace("Baseline_", "").replace("_", " + ")
                sns.kdeplot(model_samples, label=model_n, linewidth=2, alpha=0.7)
        prior_dist = priors.get(var_name, None)
        if prior_dist is not None:
            baseline_samples = samples[baseline_index].get(str(var_name), None)
            if baseline_samples is not None:
                x_vals = jnp.linspace(min_val, max_val, 200)
                prior_pdf = jnp.exp(prior_dist.log_prob(x_vals))
                ax.plot(
                    x_vals,
                    prior_pdf,
                    color="orange",
                    linestyle="--",
                    linewidth=2,
                    label="Prior",
                )
        if actual_val is not None:
            ax.axvline(actual_val, color="red", linestyle="--", linewidth=2, label="GT")
        ax.set_xlabel(var_name)
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_histogram_{var_name}.png", dpi=200)
        plt.clf()
        plt.close(fig)


def plot_models_predictive_means(
    f_hats, map_data, save_path: Path, obs_mask: Union[jax.Array, bool] = True, log=True
):
    std_vals = [f.std(axis=0) for f in f_hats[1:]]
    f_hats_plot = f_hats.copy()
    if log:
        f_hats_plot = [jnp.log(f_mean + 1) for f_mean in f_hats_plot]
    observed_y = f_hats_plot[0]
    true_y = f_hats_plot[0]
    n_models = len(f_hats_plot) - 1
    if not isinstance(obs_mask, bool):
        observed_y = np.ma.masked_where(~obs_mask, observed_y)
    mean_vals = [observed_y, true_y] + [f.mean(axis=0) for f in f_hats_plot[1:]]
    vmin_mean = float(jnp.min(jnp.array([m.min() for m in mean_vals])))
    vmax_mean = float(jnp.max(jnp.array([m.max() for m in mean_vals])))
    vmin_std = float(jnp.min(jnp.array([s.min() for s in std_vals])))
    vmax_std = float(jnp.max(jnp.array([s.max() for s in std_vals])))
    n_cols = 2 + n_models * 2
    fig, ax = plt.subplots(
        1,
        n_cols,
        figsize=(4 * n_cols, 9),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    plot_on_map(
        ax[0], map_data, observed_y, vmin=vmin_mean, vmax=vmax_mean, legend=False
    )
    ax[0].set_title("Observed y")
    ax[0].set_axis_off()
    plot_on_map(ax[1], map_data, true_y, vmin=vmin_mean, vmax=vmax_mean, legend=False)
    ax[1].set_title("True y")
    ax[1].set_axis_off()
    for i in range(n_models):
        mean_i = f_hats_plot[i + 1].mean(axis=0)
        std_i = std_vals[i]
        col_mean = 2 + i * 2
        col_std = 2 + i * 2 + 1
        last_model = i == n_models - 1
        plot_on_map(
            ax[col_mean], map_data, mean_i, vmin=vmin_mean, vmax=vmax_mean, legend=False
        )
        ax[col_mean].set_title(r"Mean $\hat{y}$")
        ax[col_mean].set_axis_off()
        if last_model:
            sm = ScalarMappable(
                norm=Normalize(vmin=vmin_mean, vmax=vmax_mean), cmap="viridis"
            )
            cb = fig.colorbar(sm, ax=ax[col_mean], shrink=0.35)
            ticks = np.linspace(vmin_mean, vmax_mean, 5)
            cb.set_ticks(ticks)
            cb.set_ticklabels([f"{np.exp(t) - 1:.0f}" for t in ticks])
        plot_on_map(
            ax[col_std],
            map_data,
            std_i,
            vmin=vmin_std,
            vmax=vmax_std,
            cmap="magma",
            legend=False,
        )
        ax[col_std].set_title("Uncertainty (Std)")
        ax[col_std].set_axis_off()
        if last_model:
            sm = ScalarMappable(
                norm=Normalize(vmin=vmin_std, vmax=vmax_std), cmap="magma"
            )
            fig.colorbar(sm, ax=ax[col_std], shrink=0.35)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close(fig)
