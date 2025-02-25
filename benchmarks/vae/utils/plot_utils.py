from datetime import datetime
from typing import Callable, Generator, Optional

import arviz as az
import flax.linen as nn
import geopandas as gpd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jax.scipy.stats import norm

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


def plot_infer_observed_coverage(post, map_data, hdi_prob=0.95, log=True):
    obs_idxs, f, f_hat = post["obs_idxs"], post["f"], post["obs"]
    f_hat_mean, f_hat_std = f_hat.mean(axis=0), f_hat.std(axis=0)
    if log:
        f, f_hat_mean = jnp.log(f + 1), jnp.log(f_hat_mean + 1)
    vmin, vmax = min(f.min(), f_hat_mean.min()), max(f.max(), f_hat_mean.max())
    fig, ax = plt.subplots(1, 5, figsize=(30, 10))
    log_str = " (Log scale)" if log else ""
    plot_on_map(ax[0], map_data, f, vmin, vmax, f"y obs{log_str}")
    plot_on_map(ax[1], map_data, f_hat_mean, vmin, vmax, f"Mean MCMC Samples{log_str}")
    plot_on_map(ax[2], map_data, f_hat_std, title="MCMC STD", cmap="plasma")
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_hat_mean - z_score * f_hat_std
    f_upper = f_hat_mean + z_score * f_hat_std
    coverage = jnp.logical_and(f >= f_lower, f <= f_upper)
    coverage_pct = coverage.mean() * 100
    cvr_title = f"{hdi_prob}% Conf Coverage\nCoverage{log_str}: {coverage_pct:.2f}%"
    plot_on_map(ax[3], map_data, coverage.astype(int), title=cvr_title, cmap="coolwarm")
    obs_title = f"Obsereved Locations ({len(obs_idxs)} locations)"
    mask = jnp.array([(1 if i in obs_idxs else 0) for i in range(map_data.shape[0])])
    plot_on_map(ax[4], map_data, mask, 0.0, 1.0, obs_title, cmap="coolwarm")
    for axis in ax:
        axis.set_axis_off()
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/Sampeled vs GT {timestamp}.png"
    fig.savefig(path, dpi=125)
    wandb.log({"Sampeled vs GT": wandb.Image(path)})
    plt.clf()
    plt.close(fig)
    return path


def plot_infer_realizations(rng_plot, map_data, f_obs, post, num_samples=5, log=True):
    log_str = " (Log scale)" if log else ""
    fig, ax = plt.subplots(1, num_samples + 1, figsize=(3 * num_samples, 16))
    samples_f = post["obs"]
    # NOTE: sets the first realisation to the actual observed one
    samples_idxs = jax.random.choice(
        rng_plot, jnp.arange(samples_f.shape[0]), (num_samples,), replace=False
    )
    if log:
        f_obs, samples_f = jnp.log(f_obs + 1), jnp.log(samples_f + 1)
    vmin = min(f_obs.min(), samples_f[samples_idxs].min())
    vmax = min(f_obs.max(), samples_f[samples_idxs].max())
    plot_on_map(
        ax[0],
        map_data,
        f_obs,
        vmin=vmin,
        vmax=vmax,
        title=f"f obs{log_str}",
        legend=False,
    )
    ax[0].set_axis_off()
    for i, s_idx in enumerate(samples_idxs):
        plot_on_map(
            ax[i + 1],
            map_data,
            samples_f[s_idx],
            vmin=vmin,
            vmax=vmax,
            title=f"Realisation {i}{log_str}",
            legend=False,
        )
        ax[i + 1].set_axis_off()

    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/Inferred Realisations {timestamp}.png"
    fig.savefig(path, dpi=250)
    wandb.log({f"Inferred Realisations{log_str}": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


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


def plot_covariance(samples, conditionals, model_name, kernel, s):
    if kernel.__name__ == "periodic":
        K = kernel(
            s, s, conditionals["var"], conditionals["ls"], conditionals["period"]
        )
    else:
        K = kernel(s, s, conditionals["var"], conditionals["ls"])
    mu_covariance = np.cov(samples["mu"], rowvar=False)
    vmin = min(K.min(), mu_covariance.min())
    vmax = max(K.max(), mu_covariance.max())
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    plot_matrix_with_colorbar(fig, ax[0, 0], K, "GT Kernel - scaled", vmin, vmax)
    plot_matrix_with_colorbar(
        fig, ax[0, 1], mu_covariance, "Inferred covariance - scaled", vmin, vmax
    )
    plot_matrix_with_colorbar(fig, ax[1, 0], K, "GT Kernel")
    plot_matrix_with_colorbar(fig, ax[1, 1], mu_covariance, "Inferred covariance")
    cond_str = ", ".join([f"{k}: {v[0]:.2f}" for k, v in conditionals.items()])
    plt.title(f"Covariance Matrix for {model_name}: {cond_str}")
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/covariance_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=150)
    wandb.log({"Covariance Matrix": wandb.Image(path)})
    plt.clf()
    plt.close(fig)


def plot_trace(samples, mcmc, conditionals):
    mcmc = az.from_numpyro(mcmc)
    var_names = [
        str(c) for c in conditionals.keys() if c in mcmc.posterior.data_vars.variables
    ]
    if len(var_names) > 0:
        az.plot_trace(mcmc, var_names=var_names)
        conditional_means = {c: samples[str(c)].mean().item() for c in var_names}
        axes = plt.gcf().get_axes()
        for i, (name, mean_val) in enumerate(conditional_means.items()):
            axes[(i * 2) + 1].set_title(f"{name} (mean: {mean_val:.2f})", fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        timestamp = datetime.now().isoformat()
        path = f"/tmp/trace_{timestamp}.png"
        plt.savefig(path, dpi=150)
        wandb.log({"Trace Plot": wandb.Image(str(path))})
        plt.clf()


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


def plot_posterior_predictives_map_points(
    gdf: gpd.GeoDataFrame,
    x_norm_vars: tuple,
    y_norm_vars: tuple,
    s_ctx: jax.Array,
    valid_lens_ctx: jax.Array,
    f_test: jax.Array,
    valid_lens_test: jax.Array,
    f_mu: jax.Array,
    f_std: jax.Array,
    sampling_policy: str,
    hdi_prob: float = 0.95,
    num_plots: int = 10,
):
    """Plots posterior predictives for geoms on the map, and saves figures."""
    num_geoms = len(gdf.geometry)
    paths = []
    x_mean, x_std = x_norm_vars
    y_mean, y_std = y_norm_vars
    for i in range(num_plots):
        v_ctx = valid_lens_ctx[i]
        s_ctx_i = s_ctx[i, :v_ctx].squeeze()
        v_test = valid_lens_test[i]
        f_test_i = f_test[i, -num_geoms:v_test].squeeze()
        f_mu_i = f_mu[i, -num_geoms:v_test].squeeze()
        f_std_i = f_std[i, -num_geoms:v_test].squeeze()

        vmin = min(f_test_i.min(), f_mu_i.min())
        vmax = max(f_test_i.max(), f_mu_i.max())

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        plot_on_map(ax[0, 0], gdf, f_test_i, vmin, vmax, "Ground Truth", "viridis")
        plot_on_map(ax[0, 1], gdf, f_mu_i, vmin, vmax, "Predicted Values", "viridis")
        context_points = np.array(
            [(s_ctx_i[:, 0] * x_std) + x_mean, (s_ctx_i[:, 1] * y_std) + y_mean]
        ).T
        plot_locations_map(
            ax[0, 2], gdf, context_points, sampling_policy, plot_blank=True
        )
        plot_on_map(ax[1, 0], gdf, f_std_i, title="Uncertainty - STD", cmap="plasma")

        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        f_lower, f_upper = f_mu_i - z_score * f_std_i, f_mu_i + z_score * f_std_i

        coverage = jnp.logical_and(f_test_i >= f_lower, f_test_i <= f_upper)
        coverage_pct = coverage.mean() * 100
        plot_on_map(
            ax[1, 1],
            gdf,
            coverage.astype(int),
            title=f"1-0 Coverage for 95% Conf\nCoverage: {coverage_pct:.2f}%",
            cmap="coolwarm",
        )
        for axis_row in ax:
            for axis in axis_row:
                axis.set_axis_off()

        plt.tight_layout()

        timestamp = datetime.now().isoformat()
        title = f"Sample {i} (GT, Prediction, Uncertainty, Coverage)"
        paths.append(f"/tmp/Meta Reg {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.9)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def log_posterior_map_predictive_plots(gdf: gpd.GeoDataFrame, sampling_policy: str):
    x_norm_vars, y_norm_vars = get_norm_vars(gdf)

    def log_posterior_predictive_plots(
        step: int,
        rng_step: int,
        state,
        batch: tuple,
        num_plots: int = 10,
    ):
        rng_dropout, rng_extra = jax.random.split(rng_step)
        (
            s_ctx,
            f_ctx,
            valid_lens_ctx,
            s_test,
            f_test,
            valid_lens_test,
            _,
            _,
            _,
        ) = batch

        f_mu, f_std, *_ = state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        paths = plot_posterior_predictives_map_points(
            gdf,
            x_norm_vars,
            y_norm_vars,
            s_ctx,
            valid_lens_ctx,
            f_test,
            valid_lens_test,
            f_mu,
            f_std,
            sampling_policy,
            num_plots=num_plots,
        )
        wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})

    return log_posterior_predictive_plots


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


def plot_vae_reconstruction_samples(
    rng: jax.Array,
    gdf: gpd.GeoDataFrame,
    x_norm_vars: tuple,
    y_norm_vars: tuple,
    state: TrainState,
    decode_only: bool,
    loader: Generator,
    conds_names: list[str],
    s: jax.Array,
    num_plots: int = 10,
    samples_per_plot: int = 3,
    kwargs={},
):
    """Plots VAE predictions on map"""
    paths = []
    x_mean, x_std = x_norm_vars
    y_mean, y_std = y_norm_vars
    for i in range(num_plots):
        rng_drop, rng_extra, rng = jax.random.split(rng, 3)
        f, z, conditionals = next(loader)
        f_hat = state.apply_fn(
            {"params": state.params, **state.kwargs},
            z if decode_only else f,
            conditionals,
            **kwargs,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        f_hat = f_hat if decode_only else f_hat[0]
        fig, ax = plt.subplots(1, samples_per_plot * 2 + 1, figsize=(20, 5))
        for i in range(samples_per_plot):
            f_i = f[i].squeeze()
            f_hat_i = f_hat[i].squeeze()
            vmin = min(f_i.min(), f_hat_i.min())
            vmax = max(f_i.max(), f_hat_i.max())
            plot_on_map(ax[2 * i], gdf, f_i, vmin, vmax, "Ground Truth", "viridis")
            plot_on_map(ax[2 * i + 1], gdf, f_hat_i, vmin, vmax, "Predicted", "viridis")
        context_points = np.array(
            [(s[:, 0] * x_std) + x_mean, (s[:, 1] * y_std) + y_mean]
        ).T
        plot_locations_map(ax[-1], gdf, context_points, plot_blank=True)
        for axis in ax:
            axis.set_axis_off()

        plt.tight_layout()

        timestamp = datetime.now().isoformat()
        title = f"Sample {i} {conds_to_title(conds_names, conditionals)}"
        paths.append(f"/tmp/VAE_Rec {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.85)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def plot_vae_decoder_samples(
    rng: jax.Array,
    gdf: gpd.GeoDataFrame,
    loader: Generator,
    conds_names: list[str],
    z_dim: int,
    surrogate_decoder: Callable,
    num_batches: int = 5,
    num_plots: int = 5,
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
    return paths


def plot_dist_analysis_plots(
    rng: jax.Array,
    loader: Generator,
    conds_names: list[str],
    z_dim: int,
    surrogate_decoder: Callable,
    map_data: gpd.GeoDataFrame,
    num_batches: int = 5,
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
        plot_kl_on_map(f_batch, f_hat, map_data, conds_str)
    return paths


def plot_vae_scatter_comp(
    rng_scatter,
    f,
    f_hat,
    conditionals,
    conds_names,
    num_samples=5,
    num_LTAs=None,
):
    paths = []
    fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    for i, ax in enumerate(axes.flatten() if num_samples > 1 else [axes]):
        rng_scatter, _ = jax.random.split(rng_scatter)
        f_i, f_hat_i = f[i], f_hat[i]
        if num_LTAs is not None:
            idxs = jax.random.choice(
                rng_scatter, f_i.shape[0], (num_LTAs,), replace=False
            )
            f_i = f_i[idxs]
            f_hat_i = f_hat_i[idxs]
        ax.scatter(f_i, f_hat_i, alpha=0.6, label="Samples")
        min_val = min(f_i.min(), f_hat_i.min())
        max_val = max(f_i.max(), f_hat_i.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y = x")
        ax.set_xlabel("True values (f_i)")
        ax.set_ylabel("Predicted values (f_hat_i)")
        ax.set_title(f"Y vs Y hat - sample {i + 1}")
        ax.legend()
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    title = f"Y vs Y hat {conds_to_title(conds_names, conditionals)}"
    paths.append(f"/tmp/VAE_Scatter {timestamp} - {title}.png")
    fig.subplots_adjust(top=0.86)
    fig.savefig(paths[-1], dpi=125)
    plt.clf()
    plt.close(fig)
    return paths


def log_vae_map_plots(
    gdf: gpd.GeoDataFrame,
    s: jax.Array,
    conds_names: list[str],
    z_dim: int,
    large_batch_loader: Generator,
    is_decoder_only: bool,
):
    x_norm_vars, y_norm_vars = get_norm_vars(gdf)

    def log_posterior_predictive_plots(
        step: int,
        rng_step: int,
        state: TrainState,
        model: nn.Module,
        loader: Generator,
        **kwargs,
    ):
        rng_drop, rng_extra, rng_dec, rng_scat, rng_dist, rng_rcn = jax.random.split(
            rng_step, 6
        )
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
        paths_rec = plot_vae_reconstruction_samples(
            rng_rcn,
            gdf,
            x_norm_vars,
            y_norm_vars,
            state,
            is_decoder_only,
            loader,
            conds_names,
            s,
            kwargs=kwargs,
        )
        paths_decoder = plot_vae_decoder_samples(
            rng_dec,
            gdf,
            loader,
            conds_names,
            z_dim,
            surrogate_decoder,
            **kwargs,
        )
        paths_scatter = plot_vae_scatter_comp(
            rng_scat,
            f,
            f_hat,
            conditionals,
            conds_names,
        )
        path_dist = plot_dist_analysis_plots(
            rng_dist,
            large_batch_loader,
            conds_names,
            z_dim,
            surrogate_decoder,
            gdf,
            **kwargs,
        )
        wandb.log({f"Scatter {step}": [wandb.Image(p) for p in paths_scatter]})
        wandb.log({f"Reconstruction {step}": [wandb.Image(p) for p in paths_rec]})
        wandb.log({f"Decoder {step}": [wandb.Image(p) for p in paths_decoder]})
        wandb.log({f"Distribution {step}": [wandb.Image(p) for p in path_dist]})

    return log_posterior_predictive_plots


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
