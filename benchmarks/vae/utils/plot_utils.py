from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import arviz as az
import flax.linen as nn
import geopandas as gpd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import seaborn as sns
from jax.scipy.stats import norm
from numpyro.infer import MCMC
from omegaconf import DictConfig

import wandb
from dl4bi.vae.train_utils import TrainState, generate_surrogate_decoder
from utils.map_utils import get_norm_vars


def plot_EB_scatter_conditionals(
    real_conds,
    inferred_conds,
    conds_names,
):
    num_conds = len(conds_names)
    fig, axes = plt.subplots(1, num_conds, figsize=(5 * num_conds, 5))
    for i, ax in enumerate(axes.flatten() if num_conds > 1 else [axes]):
        c_name, real_c, inf_c = conds_names[i], real_conds[:, i], inferred_conds[:, i]
        ax.scatter(real_c, inf_c, alpha=0.6)
        min_val = min(real_c.min(), inf_c.min())
        max_val = max(real_c.max(), inf_c.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
        ax.set_xlabel("True values")
        ax.set_ylabel("EB approx values")
        ax.set_title(f"real {c_name} vs {c_name} hat - sample {i + 1}")
        ax.legend()
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    plot_path = f"/tmp/EB_Scatter {timestamp} - Conds vs Conds hat.png"
    fig.subplots_adjust(top=0.86)
    fig.savefig(plot_path, dpi=125)
    plt.clf()
    plt.close(fig)
    return plot_path


def plot_infer_map_sum(post, map_data, log=True):
    obs_idxs, f, f_hat = post["obs_idxs"], post["f"], post["obs"]
    f_hat_mean, f_hat_std = f_hat.mean(axis=0), f_hat.std(axis=0)
    if log:
        f, f_hat_mean = jnp.log(f + 1), jnp.log(f_hat_mean + 1)
    vmin, vmax = min(f.min(), f_hat_mean.min()), max(f.max(), f_hat_mean.max())
    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    log_str = " (Log scale)" if log else ""
    plot_on_map(ax[0], map_data, f, vmin, vmax, f"y obs{log_str}")
    plot_on_map(ax[1], map_data, f_hat_mean, vmin, vmax, f"Mean MCMC Samples{log_str}")
    plot_on_map(ax[2], map_data, f_hat_std, title="MCMC STD", cmap="plasma")
    obs_title = f"Obsereved Locations ({len(obs_idxs)} locations)"
    mask = jnp.array([(1 if i in obs_idxs else 0) for i in range(map_data.shape[0])])
    plot_on_map(ax[3], map_data, mask, 0.0, 1.0, obs_title, cmap="coolwarm")
    for axis in ax:
        axis.set_axis_off()
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/Infer summary {timestamp}.png"
    fig.savefig(path, dpi=125)
    wandb.log({"Infer summaryT": wandb.Image(path)})
    plt.clf()
    plt.close(fig)
    return path


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
    if prev_real is None and inference_model == "binomial" and population is not None:
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
        ax.set_title(f"{val_n} vs. {model.replace('_', ' ')} mean {val_n.lower()}")
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
    if prev_real is None and inference_model == "binomial" and population is not None:
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
        1, len(prev_hat_means), figsize=(6 * len(prev_hat_means), 8)
    )
    log_str = " (Log scale)" if log else ""
    for i, prev_mean in enumerate(prev_hat_means):
        title = f"{models[i - 1]}: Mean {val_n.lower()}"
        if i == 0 and use_real:
            title = f"Observed {val_n.lower()}"
        ax = axes if len(prev_hat_means) == 1 else axes[i]
        plot_on_map(ax, map_data, prev_mean, vmin, vmax, f"{title}{log_str}")
        ax.set_axis_off()
    plt.tight_layout()
    if save_path is None:
        save_path = f"/tmp/Observed prevalence{datetime.now().isoformat()}.png"
    fig.savefig(save_path, dpi=125)
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


def plot_histograms(samples, conditionals, priors):
    num_plots = len(conditionals)
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 4))
    for i, (name, actual_val) in enumerate(conditionals.items()):
        ax = axes[i]
        if str(name) in samples:
            sample_values = samples[str(name)]
            ax.hist(
                sample_values,
                bins=20,
                color="skyblue",
                edgecolor="black",
                label="Posterior Samples",
            )
            prior_dist = priors[name]
            x_vals = jnp.linspace(min(sample_values), max(sample_values), 100)
            prior_pdf = jnp.exp(prior_dist.log_prob(x_vals))
            ax_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            ax.plot(
                x_vals,
                prior_pdf * len(sample_values) * ax_range / 20,
                color="orange",
                linestyle="--",
                linewidth=1,
                label="Prior Distribution",
            )
        if actual_val is not None:
            ax.axvline(
                actual_val,
                color="red",
                linestyle="--",
                linewidth=1,
                label="True Value",
            )
        title = f"{name}: {actual_val:.2f}" if actual_val is not None else name
        ax.set_title(title)
        ax.set_xlabel(name)
        ax.legend()

    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/histograms_{timestamp}.png"
    plt.savefig(path, dpi=150)
    wandb.log({"Histograms for Conditionals": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


def plot_infer_grid_sum(rng, post, hdi_prob=0.95, num_decodings=10):
    s_flat = post["s"].squeeze()
    f, f_hat = post["f"], post["obs"]
    f_hat_mean, f_hat_std = f_hat.mean(axis=0), f_hat.std(axis=0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 8))

    idxs = jax.random.choice(
        rng, jnp.arange(f_hat.shape[0]), shape=(num_decodings,), replace=False
    )
    ax[0].plot(s_flat, f_hat_mean, color="red", label=f"mean {r'$\hat{f}$'}")
    ax[0].plot(s_flat, f.squeeze(), color="black", label=r"$f$")
    ax[0].set_title("Postetior predictive samples")
    ax[0].legend()
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_hat_mean - z_score * f_hat_std
    f_upper = f_hat_mean + z_score * f_hat_std
    coverage = jnp.logical_and(f >= f_lower, f <= f_upper)
    coverage_pct = coverage.mean() * 100
    ax[1].plot(s_flat, f.squeeze(), color="black", label=r"$f$")
    ax[1].plot(s_flat, f_hat_mean, color="red", label=f"mean {r'$\hat{f}$'}")
    ax[1].fill_between(
        s_flat,
        f_lower,
        f_upper,
        color="red",
        alpha=0.3,
        label=f"{hdi_prob * 100:.0f}% HDI",
    )
    ax[1].set_title(f"Coverage: {coverage_pct:.1f}%")
    ax[1].legend()
    for j in idxs:
        ax[2].plot(s_flat, f_hat[j].squeeze(), label=f"{r'$\hat{f}$'} - {j}")
    ax[2].set_title("Postetior predictive samples")

    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/Infer summary {timestamp}.png"
    fig.savefig(path, dpi=125)
    wandb.log({"Infer summaryT": wandb.Image(path)})
    plt.clf()
    plt.close(fig)
    return path


def plot_inference_run(
    rng: jax.Array,
    inference_model: DictConfig,
    model_name: str,
    hmc_res: tuple[dict, MCMC, dict],
    f_obs: jax.Array,
    surrogate_conds: dict,
    priors: dict,
    map_data: Optional[gpd.GeoDataFrame],
    log_scale_plots: bool,
):
    samples, mcmc, post = hmc_res
    if map_data is not None:
        plot_infer_map_sum(post, map_data, log=log_scale_plots)
        plot_map_predictive(rng, f_obs, post["obs"], map_data, log=log_scale_plots)
    elif post["s"].shape[-1] == 1:
        plot_infer_grid_sum(rng, post)
    # NOTE: in case the chains\samples are very tightly batch (not converged) the plotting will fail
    try:
        plot_infer_trace(samples, mcmc, surrogate_conds)
        plot_histograms(samples, surrogate_conds, priors)
    except ValueError:
        pass
    plot_vae_scatter_plot(
        f_obs[None, ...],
        post["obs"].mean(axis=0)[None, ..., None],
        [it for _, it in surrogate_conds.items()],
        list(surrogate_conds.keys()),
        num_samples=1,
    )
    if inference_model.model.func not in ["poisson", "binomial"]:
        return
    prev_real, population = None, None
    if map_data is not None and "population" in map_data.columns:
        population = map_data.population
    if map_data is None or "data" not in map_data.columns:
        prev_real = post["beta"] + post["spatial_eff"]
    prev_hat = samples["beta"][..., None] + samples["mu"]
    if map_data is not None:
        plot_models_mean_prevalence(
            prev_real,
            [prev_hat],
            f_obs,
            population,
            [model_name],
            inference_model.model.func,
            map_data,
            population_scale=inference_model.get("population_scale", 1),
        )
    plot_prevalence_scatter_comp(
        prev_real,
        [prev_hat],
        f_obs,
        population,
        [model_name],
        inference_model.model.func,
        population_scale=inference_model.get("population_scale", 1),
    )


def plot_kl_on_map(
    f: jax.Array, f_hat: jax.Array, map_data: gpd.GeoDataFrame, conds_str: str
):
    """Plots the KL divergance between two empirical distributions for every
    location on the map. This function assumes MVN distribution

    Args:
        f (jax.Array): Samples from the true distribution
        f_hat (jax.Array): Samples from the model's distribution
        map_data (gpd.GeoDataFrame): map data frame
    """
    true_mean = jnp.mean(f, axis=0).squeeze()
    true_var = jnp.var(f, axis=0).squeeze()
    post_mean = jnp.mean(f_hat, axis=0).squeeze()
    post_var = jnp.var(f_hat, axis=0).squeeze()
    kl_per_location = (
        jnp.log(jnp.sqrt(post_var) / jnp.sqrt(true_var))
        + (true_var + (true_mean - post_mean) ** 2) / (2 * post_var)
        - 0.5
    )
    kl_total = jnp.mean(kl_per_location)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plot_on_map(
        ax,
        map_data,
        kl_per_location,
        0.0,
        kl_per_location.max().item(),
        f"KL diveregence per location (Mean KL: {kl_total:.2f}, {conds_str})",
        "coolwarm",
    )
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/KL divergence {timestamp}.png"
    fig.savefig(path, dpi=125)
    wandb.log({"KL divergence": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


def plot_vae_reconstruction(
    rng: jax.Array,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    state: TrainState,
    model: str,
    loader,
    conds_names: list[str],
    is_decoder_only: bool,
    save_dir: Optional[Path] = None,
    num_plots: int = 5,
    samples_per_plot: int = 3,
    plot_locations=False,
    step="",
    **kwargs,
):
    """Plots VAE predictions on map"""
    x_norm_vars, y_norm_vars = get_norm_vars(map_data)
    x_mean, x_std = x_norm_vars
    y_mean, y_std = y_norm_vars
    paths = []
    for i in range(num_plots):
        rng_drop, rng_extra, rng = jax.random.split(rng, 3)
        f, z, conditionals = next(loader)
        f_hat = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z if model == "DeepRV" else f,
            conditionals,
            **kwargs,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        # NOTE: full VAEs architectures return additional outputs
        f_hat = f_hat if is_decoder_only else f_hat[0]
        fig, ax = plt.subplots(
            1, samples_per_plot * 2 + int(plot_locations), figsize=(16, 5)
        )
        for j in range(samples_per_plot):
            f_j = f[j].squeeze()
            f_hat_j = f_hat[j].squeeze()
            vmin = min(f_j.min(), f_hat_j.min())
            vmax = max(f_j.max(), f_hat_j.max())
            plot_on_map(ax[2 * j], map_data, f_j, vmin, vmax, r"$f$", "viridis")
            plot_on_map(
                ax[2 * j + 1], map_data, f_hat_j, vmin, vmax, r"$\hat{f}$", "viridis"
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


def plot_vae_decoder_samples(
    rng: jax.Array,
    gdf: gpd.GeoDataFrame,
    loader: Generator,
    conds_names: list[str],
    z_dim: int,
    surrogate_decoder: Callable,
    num_batches: int = 5,
    num_plots: int = 5,
    step="",
    **kwargs,
):
    paths = []
    for i in range(num_batches):
        rng_z, rng = jax.random.split(rng)
        fig, ax = plt.subplots(1, num_plots + 1, figsize=(5 * num_plots, 5))
        f_batch, _, conditionals = next(loader)
        f = f_batch[0]
        z = jax.random.normal(rng_z, shape=(f_batch.shape[0], z_dim))
        f_hat = surrogate_decoder(z, conditionals, **kwargs)
        vmin = min(f.min(), f_hat.min())
        vmax = max(f.max(), f_hat.max())
        plot_on_map(ax[0], gdf, f, vmin, vmax, "GT sample", "viridis")
        for j in range(num_plots):
            plot_on_map(
                ax[j + 1], gdf, f_hat[j], vmin, vmax, f"Realisation {j + 1}", "viridis"
            )
        for axis in ax:
            axis.set_axis_off()
        plt.tight_layout()
        timestamp = datetime.now().isoformat()
        title = f"Sample {i} {conds_to_title(conds_names, conditionals)}"
        paths.append(f"/tmp/VAE_Decoder {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.86)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    wandb.log({f"Decoder {step}": [wandb.Image(p) for p in paths]})


def plot_violin(post, f_batch, model_name, num_locations=10, log_scale: bool = False):
    obs_idxs = post["obs_idxs"]
    random_idxs = np.random.choice(post["obs"].shape[1], num_locations, replace=False)
    obs_data = [post["obs"][:, idx] for idx in random_idxs]
    true_data = f_batch[:, random_idxs]
    if log_scale:
        obs_data = [jnp.log(obs_d + 1) for obs_d in obs_data]
        true_data = jnp.log(true_data + 1)
    data = []
    for i, (obs, true) in enumerate(zip(obs_data, true_data.T)):
        obs_str = " obs" if random_idxs[i] in obs_idxs else ""
        location = f"Loc {random_idxs[i]}{obs_str}"
        data.extend([(value.item(), location, "True Data") for value in true])
        data.extend([(value.item(), location, "Posterior Data") for value in obs])
    df = pd.DataFrame(data, columns=["Value", "Location", "Type"])
    fig = plt.figure(figsize=(16, 8))
    sns.violinplot(
        data=df,
        x="Location",
        y="Value",
        hue="Type",
        split=True,
        palette={"True Data": "blue", "Posterior Data": "red"},
        inner="quartile",
        linewidth=1.2,
    )
    title = f"True vs Sampeled Distributions {model_name}{' (log_scale)' if log_scale else ''}"
    plt.title(title, fontsize=16)
    plt.xlabel("Locations", fontsize=14)
    plt.ylabel("Observation Value", fontsize=14)
    plt.legend(title="Distribution Type", loc="upper right", fontsize=12)
    plt.xticks(rotation=45)
    timestamp = datetime.now().isoformat()
    path = f"/tmp/violin_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=200)
    wandb.log({"Violin Plot": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


def plot_matrix_with_colorbar(fig, axis, matrix, title, min_v=None, max_v=None):
    im = axis.imshow(matrix, cmap="viridis", vmin=min_v, vmax=max_v)
    axis.set_title(title)
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)


def plot_dist_analysis_plots(
    rng: jax.Array,
    loader: Generator,
    conds_names: list[str],
    z_dim: int,
    surrogate_decoder: Callable,
    map_data: Optional[gpd.GeoDataFrame],
    num_batches: int = 5,
    step="",
    **kwargs,
):
    paths = []
    for i in range(num_batches):
        rng_z, rng = jax.random.split(rng)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        f_batch, _, conditionals = next(loader)
        f = f_batch[0]
        z = jax.random.normal(rng_z, shape=(f_batch.shape[0], z_dim))
        f_hat = surrogate_decoder(z, conditionals, **kwargs)
        plot_matrix_with_colorbar(
            fig, ax[0], np.cov(f_batch.squeeze(), rowvar=False), "Empirical GT Cov"
        )
        plot_matrix_with_colorbar(
            fig, ax[1], np.cov(f_hat.squeeze(), rowvar=False), "Empirical decoder Cov"
        )
        for axis in ax:
            axis.set_axis_off()
        plt.tight_layout()
        timestamp = datetime.now().isoformat()
        conds_str = conds_to_title(conds_names, conditionals)
        title = f"Sample {i} {conds_str}"
        paths.append(f"/tmp/VAE_cov {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.86)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
        plot_violin({"obs_idxs": jnp.arange(f.shape[0]), "obs": f_hat}, f_batch, title)
        if map_data is not None:
            plot_kl_on_map(f_batch, f_hat, map_data, conds_str)
    wandb.log({f"Distribution {step}": [wandb.Image(p) for p in paths]})


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


def log_vae_map_plots(
    map_data: gpd.GeoDataFrame,
    s: jax.Array,
    conds_names: list[str],
    z_dim: int,
    large_batch_loader: Generator,
    is_decoder_only: bool,
):
    def log_plots(
        step: int,
        rng_step: int,
        state: TrainState,
        model: nn.Module,
        loader: Generator,
        **kwargs,
    ):
        rng_drop, rng_extra, rng_dec, rng_dist, rng_rcn = jax.random.split(rng_step, 5)
        f, z, conditionals = next(loader)
        f_hat = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z if is_decoder_only else f,
            conditionals,
            **kwargs,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        f_hat = f_hat if is_decoder_only else f_hat[0]
        surrogate_decoder = generate_surrogate_decoder(state, model)
        plot_vae_reconstruction(
            rng_rcn,
            s,
            map_data,
            state,
            model.__class__.__name__,
            loader,
            conds_names,
            is_decoder_only,
            plot_locations=True,
            step=str(step),
            **kwargs,
        )
        plot_vae_decoder_samples(
            rng_dec,
            map_data,
            loader,
            conds_names,
            z_dim,
            surrogate_decoder,
            step=str(step),
            **kwargs,
        )
        plot_vae_scatter_plot(
            f,
            f_hat,
            conditionals,
            conds_names,
            step=str(step),
        )
        plot_dist_analysis_plots(
            rng_dist,
            large_batch_loader,
            conds_names,
            z_dim,
            surrogate_decoder,
            map_data,
            step=str(step),
            **kwargs,
        )

    return log_plots


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
    ax.set_title(
        f"{len(locations)} Locations (ctx): {sampling_policy.replace('_', ' ')}"
    )
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


def plot_vae_rec_1d(
    rng: jax.Array,
    s: jax.Array,
    state: TrainState,
    model: str,
    loader,
    conds_names: list[str],
    is_decoder_only: bool,
    save_dir: Optional[Path] = None,
    num_plots: int = 5,
    samples_per_plot: int = 3,
    step="",
    **kwargs,
):
    """Plots VAE predictions on map"""
    paths = []
    s_flat = s.squeeze()
    for i in range(num_plots):
        rng_drop, rng_extra, rng = jax.random.split(rng, 3)
        f, z, conditionals = next(loader)
        f_hat = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z if is_decoder_only else f,
            conditionals,
            **kwargs,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        # NOTE: full VAEs architectures return additional outputs
        f_hat = f_hat if is_decoder_only else f_hat[0]
        fig, ax = plt.subplots(1, samples_per_plot, figsize=(16, 5))
        for j in range(samples_per_plot):
            ax[j].plot(s_flat, f[j].squeeze(), color="black", label=r"$f$")
            ax[j].plot(s_flat, f_hat[j].squeeze(), color="red", label=r"$\hat{f}$")
            ax[j].legend()
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


def plot_vae_decoder_1d(
    rng: jax.Array,
    s: jax.Array,
    loader: Generator,
    conds_names: list[str],
    z_dim: int,
    surrogate_decoder: Callable,
    num_batches: int = 5,
    num_decodings: int = 10,
    step="",
    **kwargs,
):
    s_flat = s.squeeze()
    fig, ax = plt.subplots(1, num_batches, figsize=(5 * num_batches, 5))
    for i in range(num_batches):
        rng_z, rng = jax.random.split(rng)
        f_batch, _, conditionals = next(loader)
        f = f_batch[0]
        z = jax.random.normal(rng_z, shape=(f_batch.shape[0], z_dim))
        f_hat = surrogate_decoder(z, conditionals, **kwargs)
        ax[i].plot(s_flat, f.squeeze(), color="black", label=r"$f$")
        for j in range(num_decodings):
            ax[i].plot(s_flat, f_hat[j].squeeze(), label=f"{r'$\hat{f}$'} - {j}")
        ax[i].set_title(f"Sample {i} {conds_to_title(conds_names, conditionals)}")
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/VAE_1d_Decoder {timestamp}.png"
    fig.subplots_adjust(top=0.86)
    fig.savefig(path, dpi=125)
    plt.clf()
    plt.close(fig)
    wandb.log({f"Decoder {step}": wandb.Image(path)})


def log_vae_grid_plots(
    map_data: Optional[gpd.GeoDataFrame],
    s: jax.Array,
    conds_names: list[str],
    z_dim: int,
    large_batch_loader: Generator,
    is_decoder_only: bool,
):
    def log_plots(
        step: int,
        rng_step: int,
        state: TrainState,
        model: nn.Module,
        loader: Generator,
        **kwargs,
    ):
        rng_drop, rng_extra, rng_dec, rng_dist, rng_rcn = jax.random.split(rng_step, 5)
        f, z, conditionals = next(loader)
        f_hat = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z if is_decoder_only else f,
            conditionals,
            **kwargs,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        f_hat = f_hat if is_decoder_only else f_hat[0]
        surrogate_decoder = generate_surrogate_decoder(state, model)
        if s.shape[-1] == 1:
            plot_vae_rec_1d(
                rng_rcn,
                s,
                state,
                model.__class__.__name__,
                loader,
                conds_names,
                is_decoder_only,
                step=str(step),
                **kwargs,
            )
            plot_vae_decoder_1d(
                rng_dec,
                s,
                loader,
                conds_names,
                z_dim,
                surrogate_decoder,
                step=str(step),
            )
        plot_vae_scatter_plot(
            f,
            f_hat,
            conditionals,
            conds_names,
            step=str(step),
        )
        plot_dist_analysis_plots(
            rng_dist,
            large_batch_loader,
            conds_names,
            z_dim,
            surrogate_decoder,
            map_data=None,
            step=str(step),
            **kwargs,
        )

    return log_plots
