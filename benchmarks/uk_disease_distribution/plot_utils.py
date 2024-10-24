from datetime import datetime

import geopandas as gpd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import norm

import wandb
from dl4bi.meta_regression.train_utils import TrainState


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

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        ax[0].set_title("Ground Truth")
        gdf["GT"] = f_test_i
        gdf.plot(
            column="GT", cmap="viridis", ax=ax[0], legend=True, vmin=vmin, vmax=vmax
        )

        ax[1].set_title("Mean Predicted Values with Ctx")
        gdf["Predicted"] = f_mu_i
        gdf.plot(
            column="Predicted",
            cmap="viridis",
            ax=ax[1],
            legend=True,
            vmin=vmin,
            vmax=vmax,
        )

        context_points = np.array(
            [(s_ctx_i[:, 0] * x_std) + x_mean, (s_ctx_i[:, 1] * y_std) + y_mean]
        ).T
        ax[1].scatter(
            context_points[:, 0],
            context_points[:, 1],
            color="red",
            marker="x",
            label="Context Points",
        )
        ax[1].legend()

        ax[2].set_title("Uncertainty - STD")
        gdf["Uncertainty"] = f_std_i
        gdf.plot(column="Uncertainty", cmap="plasma", ax=ax[2], legend=True)

        z_score = jnp.abs(norm.ppf((1 - hdi_prob) / 2))
        f_lower, f_upper = f_mu_i - z_score * f_std_i, f_mu_i + z_score * f_std_i

        coverage = jnp.logical_and(f_test_i >= f_lower, f_test_i <= f_upper)
        coverage_pct = coverage.mean() * 100
        ax[3].set_title(f"1-0 Coverage for 95% Conf\n" f"Coverage: {coverage_pct:.2f}%")
        gdf["Coverage"] = coverage.astype(int)  # 1 if within interval, 0 if not

        gdf.plot(column="Coverage", cmap="coolwarm", ax=ax[3], legend=True)

        for axis in ax:
            axis.set_axis_off()

        plt.tight_layout()

        timestamp = datetime.now().isoformat()
        title = f"Sample {i} (GT, Prediction, Uncertainty, Coverage)"
        paths.append(f"/tmp/Meta Reg {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def get_norm_vars(gdf: gpd.GeoDataFrame):
    centroid_x = gdf.geometry.centroid.x
    centroid_y = gdf.geometry.centroid.y
    return (centroid_x.mean(), centroid_x.std()), (centroid_y.mean(), centroid_y.std())


def log_posterior_map_predictive_plots(gdf: gpd.GeoDataFrame):
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
            num_plots=num_plots,
        )
        wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})

    return log_posterior_predictive_plots


def plot_vae_map_points(
    gdf: gpd.GeoDataFrame,
    f: jax.Array,
    f_hat: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    num_plots: int = 10,
):
    """Plots posterior predictives for geoms on the map, and saves figures."""
    paths = []
    for i in range(num_plots):
        f_i = f[i].squeeze()
        f_hat_i = f_hat[i].squeeze()

        vmin = min(f_i.min(), f_hat_i.min())
        vmax = max(f_i.max(), f_hat_i.max())

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.subplots_adjust(top=0.8)
        ax[0].set_title("Ground Truth")
        gdf["GT"] = f_i
        gdf.plot(
            column="GT", cmap="viridis", ax=ax[0], legend=True, vmin=vmin, vmax=vmax
        )

        ax[1].set_title("Predicted Values")
        gdf["Predicted"] = f_hat_i
        gdf.plot(
            column="Predicted",
            cmap="viridis",
            ax=ax[1],
            legend=True,
            vmin=vmin,
            vmax=vmax,
        )
        for axis in ax:
            axis.set_axis_off()

        plt.tight_layout()

        timestamp = datetime.now().isoformat()
        title = f"Sample {i} (var: {var[0]:0.2f}, ls: {ls[0]:0.2f})"
        paths.append(f"/tmp/VAE {timestamp} - {title}.png")
        fig.suptitle(title)
        fig.savefig(paths[-1], dpi=125)
        plt.clf()
        plt.close(fig)
    return paths


def log_vae_map_plots(gdf: gpd.GeoDataFrame):
    def log_posterior_predictive_plots(
        step: int,
        rng_step: int,
        state: TrainState,
        batch: tuple,
        num_plots: int = 10,
    ):
        rng_dropout, rng_extra = jax.random.split(rng_step)
        f, var, ls, _, *_ = batch
        f_hat, _, _ = state.apply_fn(
            {"params": state.params, **state.kwargs},
            f,
            var,
            ls,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        paths = plot_vae_map_points(
            gdf,
            f,
            f_hat,
            var,
            ls,
            num_plots=num_plots,
        )
        wandb.log({f"Step {step}": [wandb.Image(p) for p in paths]})

    return log_posterior_predictive_plots
