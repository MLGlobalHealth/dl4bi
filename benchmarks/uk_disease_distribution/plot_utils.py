from datetime import datetime
from typing import Generator, Optional

import arviz as az
import flax.linen as nn
import geopandas as gpd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import norm

import wandb
from dl4bi.meta_regression.train_utils import TrainState


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

    def plot_matrix_with_colorbar(axis, matrix, title, min_v=None, max_v=None):
        im = axis.imshow(matrix, cmap="viridis", vmin=min_v, vmax=max_v)
        axis.set_title(title)
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

    plot_matrix_with_colorbar(ax[0, 0], K, "GT Kernel - scaled", vmin, vmax)
    plot_matrix_with_colorbar(
        ax[0, 1], mu_covariance, "Inferred covariance - scaled", vmin, vmax
    )
    plot_matrix_with_colorbar(ax[1, 0], K, "GT Kernel")
    plot_matrix_with_colorbar(ax[1, 1], mu_covariance, "Inferred covariance")
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


def get_norm_vars(gdf: gpd.GeoDataFrame):
    centroid_x = gdf.geometry.centroid.x
    centroid_y = gdf.geometry.centroid.y
    return (centroid_x.mean(), centroid_x.std()), (centroid_y.mean(), centroid_y.std())


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
    conditionals_names: list[str],
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
        title = generate_title_from_conditionals(i, conditionals_names, conditionals)
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
    conditionals_names: list[str],
    z_dim: int,
    model: nn.Module,
    num_batches: int = 3,
    num_plots: int = 5,
):
    paths = []
    for i in range(num_batches):
        rng_z, _ = jax.random.split(rng_decoder, 2)
        fig, ax = plt.subplots(1, num_plots + 1, figsize=(5 * num_plots, 5))
        f, _, conditionals = next(loader)
        f = f[0]
        z = jax.random.normal(rng_z, shape=(num_plots, z_dim))
        batched_conditionals = jnp.repeat(
            jnp.stack(conditionals).reshape(1, -1), repeats=z.shape[0], axis=0
        )

        f_hat = model.decoder.apply(
            {"params": state.params["decoder"], **state.kwargs},
            jnp.hstack([z, batched_conditionals]),
        ).reshape((num_plots,) + f.shape)
        vmin = min(f.min(), f_hat.min())
        vmax = max(f.max(), f_hat.max())
        plot_on_map(ax[0], gdf, f, vmin, vmax, "Ground Truth", "viridis")
        for j in range(num_plots):
            plot_on_map(
                ax[j + 1], gdf, f_hat[j], vmin, vmax, f"Realisation {j + 1}", "viridis"
            )
        for axis in ax:
            axis.set_axis_off()
        plt.tight_layout()
        timestamp = datetime.now().isoformat()
        title = generate_title_from_conditionals(i, conditionals_names, conditionals)
        paths.append(f"/tmp/VAE_Decoder {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.subplots_adjust(top=0.86)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def log_vae_map_plots(
    gdf: gpd.GeoDataFrame, s: jax.Array, conditionals_names: list[str], z_dim: int
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
        rng_dropout, rng_extra, rng_decoder = jax.random.split(rng_step, 3)
        f, _, conditionals = next(loader)
        f_hat, _, _ = state.apply_fn(
            {"params": state.params, **state.kwargs},
            f,
            conditionals,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        paths = plot_vae_reconstruction_samples(
            gdf,
            x_norm_vars,
            y_norm_vars,
            f,
            f_hat,
            conditionals,
            conditionals_names,
            s,
            num_plots=num_plots,
        )
        paths_decoder = plot_vae_decoder_samples(
            rng_decoder, gdf, loader, state, conditionals_names, z_dim, model
        )
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
    cmap: str = "virdis",
):
    ax.set_title(title)
    gdf["TEMP"] = values
    gdf.plot(column="TEMP", cmap=cmap, ax=ax, legend=True, vmin=vmin, vmax=vmax)


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


def generate_title_from_conditionals(
    sample_num: int, conditionals_names: list[str], conditionals: list[jax.Array]
):
    return (
        f"Sample {sample_num} ("
        + ", ".join(
            [
                f"{conditionals_names[j]}: {conditionals[j][0]:0.2f}"
                for j in range(len(conditionals_names))
            ]
        )
        + ")"
    )
