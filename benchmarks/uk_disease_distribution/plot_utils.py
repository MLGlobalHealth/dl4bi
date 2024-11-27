from datetime import datetime
from typing import Generator, Optional

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
from map_utils import get_norm_vars

import wandb
from dl4bi.meta_regression.train_utils import TrainState


def plot_infer_observed_coverage(post, map_data, model_name, hdi_prob=0.95, log=True):
    obs_idxs, f, f_hat = post["obs_idxs"], post["f"], post["obs"]
    vmin, vmax = min(f.min(), f_hat.min()), max(f.max(), f_hat.max())
    f_hat_mean, f_hat_std = f_hat.mean(axis=0), f_hat.std(axis=0)
    fig, ax = plt.subplots(1, 5, figsize=(30, 10))
    plot_on_map(ax[0], map_data, f, vmin, vmax, "y obs - noised", "viridis")
    plot_on_map(ax[1], map_data, f_hat_mean, vmin, vmax, "Mean MCMC Samples", "viridis")
    plot_on_map(ax[2], map_data, f_hat_std, title="MCMC STD", cmap="plasma")
    z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
    f_lower = f_hat_mean - z_score * f_hat_std
    f_upper = f_hat_mean + z_score * f_hat_std
    coverage = jnp.logical_and(f >= f_lower, f <= f_upper)
    coverage_pct = coverage.mean() * 100
    cvr_title = f"1-0 Coverage for {hdi_prob}% Conf\n" f"Coverage: {coverage_pct:.2f}%"
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
    if log:
        wandb.log({f"Sampeled vs GT - {model_name}": wandb.Image(path)})
    plt.clf()
    return path


def plot_infer_realizations(
    rng_plot, map_data, f_batch, post, model_name, num_samples=10
):
    fig, ax = plt.subplots(2, num_samples, figsize=(3 * num_samples, 16))
    samples_f = post["obs"]
    rng_true, rng_samples = jax.random.split(rng_plot)
    true_idxs = jax.random.choice(
        rng_true, jnp.arange(f_batch.shape[0]), (num_samples,), replace=False
    )
    # NOTE: sets the first realisation to the actual observed one
    true_idxs = true_idxs.at[0].set(0)
    samples_idxs = jax.random.choice(
        rng_samples, jnp.arange(samples_f.shape[0]), (num_samples,), replace=False
    )
    vmin = min(f_batch[true_idxs].min(), samples_f[samples_idxs].min())
    vmax = min(f_batch[true_idxs].max(), samples_f[samples_idxs].max())
    for i, (t_idx, s_idx) in enumerate(zip(true_idxs, samples_idxs)):
        plot_on_map(
            ax[0, i],
            map_data,
            f_batch[t_idx],
            vmin=vmin,
            vmax=vmax,
            title=f"True realisation {i}" if i > 0 else "y obs - noiseless",
            legend=False,
        )
        plot_on_map(
            ax[1, i],
            map_data,
            samples_f[s_idx],
            vmin=vmin,
            vmax=vmax,
            title=f"Inferred realisation {i}",
            legend=False,
        )
        ax[0, i].set_axis_off()
        ax[1, i].set_axis_off()

    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/Inferred Realisations {timestamp}.png"
    fig.savefig(path, dpi=250)
    wandb.log({f"Inferred Realisations - {model_name}": wandb.Image(path)})
    plt.clf()


def plot_kl_on_map(
    map_data: gpd.GeoDataFrame, kl_per_location: jax.Array, model_name: str
):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    plot_on_map(
        ax,
        map_data,
        kl_per_location,
        0.0,
        kl_per_location.max().item(),
        f"KL diveregence per location (Mean: {kl_per_location.mean():.2f})",
        "coolwarm",
    )
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/KL divergence {timestamp}.png"
    fig.savefig(path, dpi=125)
    wandb.log({f"KL divergence - {model_name}": wandb.Image(path)})
    plt.clf()


def plot_violin(post, f_batch, model_name, num_locations=10):
    obs_idxs = post["obs_idxs"]
    random_idxs = np.random.choice(post["obs"].shape[1], num_locations, replace=False)
    obs_data = [post["obs"][:, idx] for idx in random_idxs]
    true_data = f_batch[:, random_idxs, :].squeeze(axis=-1)
    data = []
    for i, (obs, true) in enumerate(zip(obs_data, true_data.T)):
        obs_str = " obs" if random_idxs[i] in obs_idxs else ""
        location = f"Loc {random_idxs[i]}{obs_str}"
        data.extend([(value.item(), location, "True Data") for value in true])
        data.extend([(value.item(), location, "Posterior Data") for value in obs])
    df = pd.DataFrame(data, columns=["Value", "Location", "Type"])
    plt.figure(figsize=(16, 8))
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
    plt.title(f"True vs Sampeled Distributions {model_name}", fontsize=16)
    plt.xlabel("Locations", fontsize=14)
    plt.ylabel("Observation Value", fontsize=14)
    plt.legend(title="Distribution Type", loc="upper right", fontsize=12)
    plt.xticks(rotation=45)
    timestamp = datetime.now().isoformat()
    path = f"/tmp/violin_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=200)
    wandb.log({f"Violin Plot {model_name}": wandb.Image(path)})
    plt.clf()


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

    plot_matrix_with_colorbar(ax[0, 0], K, "GT Kernel - scaled", vmin, vmax)
    plot_matrix_with_colorbar(
        ax[0, 1], mu_covariance, "Inferred covariance - scaled", vmin, vmax
    )
    plot_matrix_with_colorbar(fig, ax[1, 0], K, "GT Kernel")
    plot_matrix_with_colorbar(fig, ax[1, 1], mu_covariance, "Inferred covariance")
    cond_str = ", ".join([f"{k}: {v[0]:.2f}" for k, v in conditionals.items()])
    plt.title(f"Covariance Matrix for {model_name}: {cond_str}")
    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/covariance_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=300)
    wandb.log({f"Covariance Matrix - {model_name}": wandb.Image(path)})
    plt.clf()


def plot_trace(samples, mcmc, conditionals, obs_noise, model_name):
    var_names = [str(c) for c in conditionals.keys()] + ["sigma"]
    az.plot_trace(az.from_numpyro(mcmc), var_names=var_names)
    conditional_means = {c: samples[str(c)].mean().item() for c in var_names}
    axes = plt.gcf().get_axes()
    for i, (name, mean_val) in enumerate(conditional_means.items()):
        axes[(i * 2) + 1].set_title(f"{name} (mean: {mean_val:.2f})", fontsize=10)
    title = f"Trace for {model_name}: " + ", ".join(
        [f"{name}: {cond[0]:g}" for name, cond in conditionals.items()]
    )
    title += f" sigma: {obs_noise}"
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().isoformat()
    path = f"/tmp/trace_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=300)
    wandb.log({title: wandb.Image(str(path))})
    plt.clf()


def plot_histograms(samples, conditionals, obs_noise, model_name, priors):
    num_plots = len(conditionals) + 1
    _, axes = plt.subplots(1, num_plots, figsize=(12, 4))
    for i, (name, actual_val) in enumerate(
        {**conditionals, "sigma": [obs_noise]}.items()
    ):
        ax = axes[i]
        sample_values = samples[str(name)]
        ax.hist(
            sample_values,
            bins=20,
            color="skyblue",
            edgecolor="black",
            label="Posterior Samples",
        )
        ax.axvline(
            actual_val[0], color="red", linestyle="--", linewidth=1, label="True Value"
        )
        prior_dist = priors[name]
        x_vals = jnp.linspace(min(sample_values), max(sample_values), 100)
        prior_pdf = jnp.exp(prior_dist.log_prob(x_vals))
        ax.plot(
            x_vals,
            prior_pdf * len(sample_values) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 20,
            color="orange",
            linestyle="--",
            linewidth=1,
            label="Prior Distribution",
        )
        ax.set_title(f"{name}: {actual_val[0]:.2f}")
        ax.set_xlabel(name)
        ax.legend()

    plt.tight_layout()
    timestamp = datetime.now().isoformat()
    path = f"/tmp/histograms_{model_name}_{timestamp}.png"
    plt.savefig(path, dpi=300)
    wandb.log({f"Histograms for Conditionals - {model_name}": wandb.Image(path)})
    plt.clf()


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
            title=f"1-0 Coverage for 95% Conf\n" f"Coverage: {coverage_pct:.2f}%",
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
        state: TrainState,
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


def plot_vae_reconstruction_samples(
    gdf: gpd.GeoDataFrame,
    x_norm_vars: tuple,
    y_norm_vars: tuple,
    f: jax.Array,
    f_hat: jax.Array,
    conditionals: list[jax.Array],
    conds_names: list[str],
    s: jax.Array,
    num_plots: int = 10,
):
    """Plots VAE predictions on map"""
    paths = []
    x_mean, x_std = x_norm_vars
    y_mean, y_std = y_norm_vars
    for i in range(num_plots):
        f_i = f[i].squeeze()
        f_hat_i = f_hat[i].squeeze()

        vmin = min(f_i.min(), f_hat_i.min())
        vmax = max(f_i.max(), f_hat_i.max())

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        plot_on_map(ax[0], gdf, f_i, vmin, vmax, "Ground Truth", "viridis")
        plot_on_map(ax[1], gdf, f_hat_i, vmin, vmax, "Predicted Values", "viridis")
        context_points = np.array(
            [(s[:, 0] * x_std) + x_mean, (s[:, 1] * y_std) + y_mean]
        ).T
        plot_locations_map(ax[2], gdf, context_points, plot_blank=True)
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
    rng_decoder,
    gdf: gpd.GeoDataFrame,
    loader: Generator,
    state: TrainState,
    conds_names: list[str],
    z_dim: int,
    model: nn.Module,
    num_batches: int = 5,
    num_plots: int = 5,
):
    violin_paths = []
    paths = []
    for i in range(num_batches):
        rng_z, _ = jax.random.split(rng_decoder, 2)
        fig, ax = plt.subplots(1, num_plots + 3, figsize=(5 * num_plots, 5))
        f_batch, _, conditionals = next(loader)
        f = f_batch[0]
        z = jax.random.normal(rng_z, shape=(f_batch.shape[0], z_dim))
        batched_conditionals = jnp.repeat(
            jnp.stack(conditionals).reshape(1, -1), repeats=z.shape[0], axis=0
        )
        f_hat = model.decoder.apply(
            {"params": state.params["decoder"], **state.kwargs},
            jnp.hstack([z, batched_conditionals]),
        ).reshape((z.shape[0],) + f.shape)
        vmin = min(f.min(), f_hat.min())
        vmax = max(f.max(), f_hat.max())
        plot_on_map(ax[0], gdf, f, vmin, vmax, "GT sample", "viridis")
        for j in range(num_plots):
            plot_on_map(
                ax[j + 1], gdf, f_hat[j], vmin, vmax, f"Realisation {j + 1}", "viridis"
            )
        plot_matrix_with_colorbar(
            fig, ax[-2], np.cov(f_batch.squeeze(), rowvar=False), "Emirical GT Cov"
        )
        plot_matrix_with_colorbar(
            fig, ax[-1], np.cov(f_hat.squeeze(), rowvar=False), "Empirical decoder Cov"
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
        violin_paths.append(
            plot_violin({"obs_idxs": jnp.arange(f.shape[0]), "obs": f_hat}, f_batch, "")
        )
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
    for i, ax in enumerate(axes.flatten()):
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
):
    x_norm_vars, y_norm_vars = get_norm_vars(gdf)

    def log_posterior_predictive_plots(
        step: int,
        rng_step: int,
        state: TrainState,
        loader: Generator,
        model: nn.Module,
        num_plots: int = 10,
    ):
        rng_drop, rng_extra, rng_dec, rng_scat = jax.random.split(rng_step, 4)
        f, _, conditionals = next(loader)
        f_hat, _, _ = state.apply_fn(
            {"params": state.params, **state.kwargs},
            f,
            conditionals,
            rngs={"dropout": rng_drop, "extra": rng_extra},
        )
        paths = plot_vae_reconstruction_samples(
            gdf,
            x_norm_vars,
            y_norm_vars,
            f,
            f_hat,
            conditionals,
            conds_names,
            s,
            num_plots=num_plots,
        )
        paths_decoder = plot_vae_decoder_samples(
            rng_dec, gdf, loader, state, conds_names, z_dim, model
        )
        paths_scatter = plot_vae_scatter_comp(
            rng_scat,
            f,
            f_hat,
            conditionals,
            conds_names,
        )
        wandb.log({f"Scatter {step}": [wandb.Image(p) for p in paths_scatter]})
        wandb.log({f"Reconstruction {step}": [wandb.Image(p) for p in paths]})
        wandb.log({f"Decoder {step}": [wandb.Image(p) for p in paths_decoder]})

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
    return (
        "("
        + ", ".join(
            [
                f"{conds_names[j]}: {conditionals[j][0]:0.2f}"
                for j in range(len(conds_names))
            ]
        )
        + ")"
    )
