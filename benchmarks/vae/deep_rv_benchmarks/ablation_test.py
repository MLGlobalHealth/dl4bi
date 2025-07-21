import sys

sys.path.append("benchmarks/vae")
from datetime import datetime
from pathlib import Path
from typing import Callable

import flax.linen as nn
import geopandas as gpd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.distributions.util import cholesky_of_inverse
from shapely.affinity import scale, translate
from sps.kernels import matern_1_2, rbf
from utils.map_utils import generate_adjacency_matrix
from utils.plot_utils import conds_to_title, plot_on_map

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import MLPDeepRV, PriorCVAE, TransformerDeepRV, gMLPDeepRV
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    prior_cvae_train_step,
)


def main(init_seed=42, num_seeds=5):
    wandb.init(mode="disabled")  # NOTE: downstream function assume active wandb
    save_dir = Path("results/ablation_test/")
    save_dir.mkdir(parents=True, exist_ok=True)
    map_data = gpd.read_file("benchmarks/vae/maps/UK")
    s = gen_spatial_structure(map_data)
    L = s.shape[0]
    models = {
        "PriorCVAE": PriorCVAE(MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L),
        "DeepRV + MLP": MLPDeepRV(dims=[L, L]),
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "DeepRV + Transfomer": TransformerDeepRV(num_blks=2, dim=64),
    }
    priors = {"ls": dist.Uniform(1.0, 100.0), "alpha": dist.Beta(4.0, 1.0)}
    result = []
    seeds = random.choice(
        random.key(init_seed), jnp.arange(100), shape=(num_seeds,), replace=False
    )
    for seed in seeds:
        rng = random.key(seed)
        rng_train, rng_test, rng_plot = random.split(rng, 3)
        for kernel in [rbf, matern_1_2, car]:
            kernel_n = kernel.__name__
            if kernel_n == "car":
                loader = gen_car_dataloader(s, priors, map_data)
            else:
                loader = gen_gp_dataloader(s, priors, kernel)
            states = []
            for model_name, nn_model in models.items():
                train_time, eval_mse, state = surrogate_model_train(
                    rng_train, rng_test, loader, model_name, nn_model
                )
                states.append(state)
                result.append(
                    {
                        "model_name": model_name,
                        "train_time": train_time,
                        "Test Norm MSE": eval_mse,
                        "kernel": kernel_n,
                        "seed": seed,
                    }
                )
            if seed == seeds[0]:
                plot_reconstruction_comp(
                    rng_plot,
                    map_data,
                    kernel_n,
                    states,
                    list(models.keys()),
                    loader,
                    ["alpha" if kernel_n == "car" else "ls"],
                    save_dir,
                )
    result = pd.DataFrame(result)
    latex_table = df_to_latex_table(result)
    result.to_csv(save_dir / "res.csv")
    with open(save_dir / "latex_table.txt", "w") as ff:
        ff.write(latex_table)


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model_name: str,
    model: nn.Module,
    train_num_steps: int = 100_000,
    valid_interval: int = 25_000,
    valid_steps: int = 5_000,
):
    train_step = prior_cvae_train_step
    lr_schedule = cosine_annealing_lr(train_num_steps, 1.0e-3, 1.0e-5)
    if model_name != "PriorCVAE":
        train_step = deep_rv_train_step
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.yogi(lr_schedule))
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        train_num_steps,
        loader,
        valid_step,
        valid_interval,
        valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="norm MSE",
    )
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, valid_steps)["norm MSE"]
    return train_time, eval_mse, state


def gen_gp_dataloader(s: Array, priors: dict, kernel: Callable, batch_size=32):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: kernel(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s, var, ls)
            f = f_jit(K, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


def gen_car_dataloader(
    s: Array, priors: dict, map_data: gpd.GeoDataFrame, batch_size=32
):
    adj_mat = generate_adjacency_matrix(map_data, self_loops=False)
    D = jnp.diag(adj_mat.sum(axis=1))
    kernel_jit = jit(lambda tau, alpha: tau * (D - (alpha * adj_mat)))
    f_jit = jit(lambda K_inv, z: jnp.einsum("ij,bj->bi", cholesky_of_inverse(K_inv), z))

    @jit
    def dataloader(rng_data):
        while True:
            rng_data, rng_alpha, rng_z = random.split(rng_data, 3)
            tau = 1.0
            alpha = priors["alpha"].sample(rng_alpha)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            precision_mat = kernel_jit(tau, alpha)
            f = f_jit(precision_mat, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([alpha])}

    return dataloader


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def car():
    pass


def gen_spatial_structure(map_data: gpd.GeoDataFrame, s_max=100):
    """generates a 0-s_max range locations from the geo-locations centroids"""
    centroids = map_data.geometry.centroid
    minx, maxx = centroids.x.min(), centroids.x.max()
    miny, maxy = centroids.y.min(), centroids.y.max()
    x_tran, x_div = minx, (maxx - minx) / s_max
    y_tran, y_div = miny, (maxy - miny) / s_max

    def norm_geom(geom):
        centered_geom = translate(geom, xoff=-x_tran, yoff=-y_tran)
        normalized_geom = scale(
            centered_geom, xfact=1 / x_div, yfact=1 / y_div, origin=(0, 0)
        )
        return normalized_geom

    norm_map = map_data.copy()
    norm_map["geometry"] = norm_map.geometry.apply(norm_geom)
    centroids = norm_map.geometry.centroid
    return jnp.stack([centroids.x.values, centroids.y.values], axis=-1)


def plot_reconstruction_comp(
    rng: Array,
    map_data: gpd.GeoDataFrame,
    kernel_n: str,
    states: list,
    models: list[str],
    loader,
    conds_names: list[str],
    save_dir: Path,
    num_plots: int = 4,
):
    """Plots VAE predictions on map"""
    rng_loader, rng = random.split(rng)
    loader = loader(rng_loader)
    save_dir = save_dir / kernel_n
    save_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_plots):
        rng_drop, rng_extra, rng = random.split(rng, 3)
        batch = next(loader)
        f, conditionals = batch["f"][0], batch["conditionals"]
        fig, ax = plt.subplots(1, len(states) + 1, figsize=(16, 5))
        f_hats = jnp.array(
            [
                state.apply_fn(
                    {"params": state.params, **state.kwargs},
                    **batch,
                    rngs={"dropout": rng_drop, "extra": rng_extra},
                )
                .f_hat[0]
                .squeeze()
                for state in states
            ]
        )
        vmax, vmin = f_hats.max().item(), f_hats.min().item()
        plot_on_map(ax[0], map_data, f, vmin, vmax, r"$f$", legend=False)
        for j, model in enumerate(models):
            plot_on_map(
                ax[j + 1],
                map_data,
                f_hats[j],
                vmin,
                vmax,
                f"{model} - " r"$\hat{f}$",
                legend=False,
            )
        for axis in ax:
            axis.set_axis_off()
        plt.tight_layout()
        title = f"{conds_to_title(conds_names, conditionals)}"
        fig.suptitle(title)
        fig.subplots_adjust(top=0.85)
        fig.savefig(save_dir / f"{kernel_n}_rec_{i}.png", dpi=125)
        plt.clf()
        plt.close(fig)


def df_to_latex_table(df):
    """
    Given a DataFrame with columns: 'model_name', 'train_time', 'Test Norm MSE', 'kernel', 'seed',
    generate a LaTeX table with:
    - Rows as model_name
    - Columns as kernels, each with two subcolumns: MSE (mean ± std), Train Time (mean ± std)
    """

    # Group by model and kernel
    grouped = df.groupby(["model_name", "kernel"])

    # Compute stats
    stats = grouped.agg(
        mse_mean=("Test Norm MSE", "mean"),
        mse_std=("Test Norm MSE", "std"),
        time_mean=("train_time", "mean"),
        time_std=("train_time", "std"),
    ).reset_index()

    # Pivot to get the desired layout
    kernels = sorted(df["kernel"].unique())
    models = sorted(df["model_name"].unique())

    table_rows = []
    for model in models:
        row = [model]
        for kernel in kernels:
            subset = stats[(stats["model_name"] == model) & (stats["kernel"] == kernel)]
            if not subset.empty:
                mse = subset.iloc[0]["mse_mean"]
                mse_std = subset.iloc[0]["mse_std"]
                time = subset.iloc[0]["time_mean"]
                time_std = subset.iloc[0]["time_std"]
                row.append(f"${mse:.3f} \\pm {mse_std:.3f}$")
                row.append(f"${time:.2f} \\pm {time_std:.2f}$")
            else:
                row.extend(["-", "-"])
        table_rows.append(row)
    col_headers = ["Model"]
    for kernel in kernels:
        col_headers.append(f"\\multicolumn{{2}}{{c}}{{{kernel}}}")
    sub_headers = [""]
    for _ in kernels:
        sub_headers.extend(["MSE", "Train Time"])
    # Build LaTeX table
    latex = "\\begin{tabular}{l" + "cc" * len(kernels) + "}\n"
    latex += " & " + " & ".join(col_headers[1:]) + " \\\\\n"
    latex += " & " + " & ".join(sub_headers[1:]) + " \\\\\n"
    latex += "\\hline\n"
    for row in table_rows:
        latex += " & ".join(row) + " \\\\\n"
    latex += "\\end{tabular}"
    print(latex)
    return latex


if __name__ == "__main__":
    main()
