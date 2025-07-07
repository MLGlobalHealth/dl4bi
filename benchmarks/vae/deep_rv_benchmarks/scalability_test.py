import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union

import arviz as az
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
import seaborn as sns
from jax import Array, jit, random, vmap
from jax.scipy.linalg import solve_triangular
from numpyro import distributions as dist
from numpyro.distributions.transforms import ParameterFreeTransform
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam
from omegaconf import DictConfig
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sps.kernels import matern_1_2
from sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState,
    cosine_annealing_lr,
    estimate_flops,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.vae import ScanTransformerDeepRV, gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(seed=42, logged_priors=True):
    rng = random.key(seed)
    save_dir = Path(f"results/scalability{'_log_priors' if logged_priors else ''}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    grids = [
        build_grid([{"start": 0.0, "stop": 100.0, "num": n}])
        for n in [256, 1024, 2048, 4096]
    ]
    models = {
        "Baseline_GP": None,
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "DeepRV + ScanTransfomer": ScanTransformerDeepRV(num_blks=2, dim=64),
        "Inducing DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "Inducing Points": None,
        "ADVI": None,
    }
    model_names = list(models.keys())
    log_ls = dist.TransformedDistribution(dist.Beta(4.0, 1.0), LogScaleTransform())
    priors = {
        "ls": log_ls if logged_priors else dist.Uniform(1.0, 100.0),
        "beta": dist.Normal(),
    }
    for s in grids:
        result = []
        rng_train, rng_test, rng_infer, rng_idxs, rng_obs, rng = random.split(rng, 6)
        L = s.shape[0]
        grid_s_path = save_dir / f"grid_{L}"
        grid_s_path.mkdir(parents=True, exist_ok=True)
        y_obs = gen_y_obs(rng_obs, s)
        obs_mask = generate_obs_mask(rng_idxs, y_obs)
        poisson_llk, cond_names = inference_model(s, priors)
        num_pts = int(min(int(2 * jnp.sqrt(L).item()), sum(obs_mask)))
        poiss_inducing, u = inference_model_inducing_points(
            s, priors, obs_mask, num_pts
        )
        y_hats, all_samples = [], []
        for model_name, nn_model in models.items():
            model_path = grid_s_path / f"{model_name}"
            model_path.mkdir(parents=True, exist_ok=True)
            is_inducing = "Inducing" in model_name
            infer_model = poiss_inducing if is_inducing else poisson_llk
            train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
            infer_gflops, train_gflops, parameters = None, None, None
            max_lr, bs, train_steps = None, None, None
            if nn_model is not None:
                optimizer, max_lr, bs, train_steps = gen_train_params(model_name, s)
                wandb.init(
                    config={
                        "model_name": model_name,
                        "grid_size": L,
                        "max_lr": max_lr,
                        "batch_size": bs,
                    },
                    mode="online",
                    name=f"{model_name}",
                    project="deep_rv_optimizations",
                    reinit=True,
                )
                loader = gen_train_dataloader(s, u, priors, is_inducing, batch_size=bs)
                (
                    train_time,
                    eval_mse,
                    surrogate_decoder,
                    infer_gflops,
                    train_gflops,
                    parameters,
                ) = surrogate_model_train(
                    rng_train,
                    rng_test,
                    loader,
                    nn_model,
                    model_path,
                    optimizer,
                    train_steps,
                )
                wandb.log({"train_time": train_time, "Test Norm MSE": eval_mse})
            if model_name != "ADVI":
                samples, mcmc, post, infer_time = hmc(
                    rng_infer,
                    infer_model,
                    y_obs,
                    obs_mask,
                    model_path,
                    surrogate_decoder,
                )
                ess = az.ess(mcmc, method="mean")
                plot_infer_trace(
                    samples, mcmc, None, cond_names, model_path / "infer_trace.png"
                )
            else:
                samples, post, infer_time = advi(
                    rng_infer, infer_model, y_obs, obs_mask, model_path
                )
            y_hats.append(post["obs"])
            all_samples.append(samples)
            res = {
                "model_name": model_name,
                "max_lr": max_lr,
                "bs": bs,
                "train_steps": train_steps,
                "grid_size": L,
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "infer_time": infer_time,
                "total_time": infer_time
                if train_time is None
                else infer_time + train_time,
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "infer_flops": infer_gflops,
                "train_flops": train_gflops,
                "parameters": parameters,
            }
            res.update(
                {f"inferred {c} mean": samples[c].mean(axis=0) for c in cond_names}
            )
            res.update(
                {f"ESS {c}": ess[c].mean().item() if ess else None for c in cond_names}
            )
            with open(model_path / "single_res.pkl", "wb") as out_file:
                pickle.dump(res, out_file)
            result.append(res)
        wass_dist = posterior_wasserstein_distance(all_samples, model_names, cond_names)
        result = pd.DataFrame(result)
        for model_name in model_names:
            if model_name == "Baseline_GP":
                continue
            for n in cond_names:
                result.loc[
                    result["model_name"] == model_name, f"{n} wasserstein distance"
                ] = wass_dist[model_name][n]
        result.to_csv(grid_s_path / "res.csv")
        plot_posterior_predictive_comparisons(
            all_samples, {}, priors, model_names, cond_names, grid_s_path / "comp"
        )
        plot_models_predictive_means(
            y_obs, y_hats, obs_mask, model_names, grid_s_path / "obs_means.png"
        )
    # plot_model_scalability_metrics(result, save_dir)


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=1_000, num_warmup=4_00)
    mcmc = MCMC(nuts, num_chains=1, num_samples=10_000, num_warmup=4_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2)
    post["infer_time"] = infer_time
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    return samples, mcmc, post, infer_time


def advi(
    rng: Array,
    model: Callable,
    y: Array,
    obs_mask: Array,
    results_dir: Path,
    num_steps: int = 50_000,
):
    rng_svi, rng_pp, rng_post = random.split(rng, 3)
    guide = AutoMultivariateNormal(model)
    optimizer = Adam(step_size=0.0001)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    start = datetime.now()
    svi_result = svi.run(rng_svi, num_steps, y=y, obs_mask=obs_mask, stable_update=True)
    infer_time = (datetime.now() - start).total_seconds()
    params = svi_result.params
    samples = guide.sample_posterior(rng_pp, params, sample_shape=(40_000,))
    post = Predictive(model, samples)(rng_post)
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    return samples, post, infer_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    results_dir: Path,
    optimizer,
    train_num_steps: int = 100_000,
    valid_interval: int = 25_000,
    valid_steps: int = 5_000,
):
    train_step = deep_rv_train_step
    flop_batch = loader(rng_train).__next__()
    # NOTE: doesn't effect actual training
    rngs = {"params": rng_train, "extra": rng_test}
    kwargs = model.init(rngs, **flop_batch)
    params = kwargs.pop("params")
    state = TrainState.create(
        apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer
    )
    infer_flops, train_flops = estimate_flops(rng_train, state, train_step, flop_batch)
    parameters = nn.tabulate(model, rngs)(**flop_batch)
    parameters = int(
        parameters.split("Total Parameters: ")[-1].split(" ")[0].replace(",", "")
    )
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
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    with open(results_dir / "train_time.pkl", "wb") as out_file:
        pickle.dump({"train_time": train_time, "eval_mse": eval_mse}, out_file)
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder, infer_flops, train_flops, parameters


@partial(jit, static_argnames=["lower"])
def jit_trin_solve_func(L_T, z, lower=False):
    s_trin = partial(solve_triangular, lower=lower)
    return vmap(s_trin, in_axes=(None, 0))(L_T, z)


def gen_train_dataloader(
    s: Array, u: Array, priors: dict, inducing=False, batch_size=32
):
    s_train = u if inducing else s
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s_train.shape[0]))
            K = kernel_jit(s_train, var, ls)
            L = jnp.linalg.cholesky(K)
            if inducing:
                f = jit_trin_solve_func(L.T, z, lower=False)
            else:
                f = f_jit(L, z)
            yield {"s": s_train, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


def inference_model(s: Array, priors: dict):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        else:
            K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson, ["ls", "beta"]


def inference_model_inducing_points(
    s: Array, priors: dict, obs_mask: Array, num_points: int
):
    """Builds a poisson likelihood inference model for inducing points"""
    kmeans = KMeans(n_clusters=num_points, random_state=0)
    u = kmeans.fit(s[obs_mask]).cluster_centers_  # shape (num_points, s.shape[1])
    surr_kwargs = {"s": u}

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_su = matern_1_2(s, u, var, ls)
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, u.shape[0]))
        if surrogate_decoder is not None:
            f_u_bar = surrogate_decoder(z, jnp.array([ls]), **surr_kwargs).squeeze()
        else:
            K_uu = matern_1_2(u, u, var, ls) + 5e-4 * jnp.eye(u.shape[0])
            L_uu = jnp.linalg.cholesky(K_uu)
            f_u_bar = solve_triangular(L_uu.T, z[0], lower=False)
        f = numpyro.deterministic("mu", K_su @ f_u_bar)
        # NOTE: uncomment to perform FITC correction for marginal variances
        # K_uu_inv_K_us = jnp.linalg.solve(K_uu, K_su.T)
        # Q_ss_diag = jnp.sum(K_su * K_uu_inv_K_us.T, axis=1)
        # delta = jnp.clip(var - Q_ss_diag, 1.0e-6, jnp.inf)  # K_ss_diag = var
        # f_mu = numpyro.sample("f_mu", dist.Normal(f_mu, jnp.sqrt(delta)).to_event(1))
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing, u


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_y_obs(rng: Array, s: Array):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, 10.0, 1.0
    K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def generate_obs_mask(rng: Array, y_obs: Array, obs_ratio: float = 0.15):
    """Creates a mask which indicates to the inference model which locations to
    observe. Randomly chooses a subset of location to be observed."""
    L = y_obs.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.isin(jnp.arange(L), obs_idxs)


def plot_models_predictive_means(
    f_obs, f_hats, obs_mask, model_names, save_path: Path, log: bool = True
):
    """
    For each model (excluding 'Baseline_GP'), create a subplot with:
    - f_obs
    - Baseline_GP prediction
    - Model prediction

    The first subplot shows the observed ground truth with mask applied.
    Plots are arranged in rows with up to 5 columns.
    """
    f_hat_means = [f.mean(axis=0).flatten() for f in f_hats]
    f_obs = f_obs.flatten()
    if log:
        f_hat_means = [jnp.log1p(f) for f in f_hat_means]
        f_obs = jnp.log1p(f_obs)
    baseline_idx = model_names.index("Baseline_GP")
    baseline_pred = f_hat_means[baseline_idx]
    model_indices = [i for i, name in enumerate(model_names) if name != "Baseline_GP"]
    num_models = len(model_indices)

    cols = 5
    total_plots = num_models + 1  # +1 for masked ground truth
    rows = (total_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), sharex=True)
    axes = axes.flatten()

    for ax in axes[total_plots:]:
        ax.set_visible(False)  # Hide unused subplots

    masked_f_obs = np.ma.masked_where(~obs_mask, f_obs)
    ax = axes[0]
    ax.plot(masked_f_obs, color="black", linewidth=1.5)
    ax.set_title(r"$y_{obs}$", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)

    for plot_idx, model_idx in enumerate(model_indices, start=1):
        model_name = model_names[model_idx]
        model_pred = f_hat_means[model_idx]
        ax = axes[plot_idx]
        ax.plot(f_obs, label=r"$y$", color="black", linewidth=1.5)
        ax.plot(baseline_pred, label="GP", linestyle="--", color="blue")
        ax.plot(model_pred, label=model_name, linestyle="-", color="green")
        ax.set_title(f"{model_name}" " Mean " r"$\hat{y}$ ", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


def posterior_wasserstein_distance(
    samples: list, model_names: list[str], var_names: list[str]
):
    """
    Computes Wasserstein distance for each variable between the posterior distributions
    of each model and the baseline "Baseline_GP".
    """
    baseline_index = model_names.index("Baseline_GP")
    baseline_samples = samples[baseline_index]
    distances = {m: {} for m in model_names if m != "Baseline_GP"}
    for model_name, model_sample in zip(model_names, samples):
        if model_name == "Baseline_GP":
            continue
        for var_name in var_names:
            baseline_var_samples = baseline_samples.get(var_name)
            model_var_samples = model_sample.get(var_name)
            if baseline_var_samples is not None and model_var_samples is not None:
                dist = wasserstein_distance(baseline_var_samples, model_var_samples)
                distances[model_name][var_name] = dist
            else:
                distances[model_name][var_name] = jnp.nan
    return distances


def gen_train_params(model_name, s, default_bs=32):
    L = s.shape[0]
    default_steps = 200_000
    max_lr = {
        "Inducing DeepRV + gMLP": 5e-3,
        "DeepRV + gMLP": 5e-3 if L <= 1024 else 1e-2,
        "DeepRV + ScanTransfomer": 1e-4,
        "DeepRV + DKA": 5e-3,
    }[model_name]
    bs = {
        "Inducing DeepRV + gMLP": default_bs,
        "DeepRV + gMLP": int(min(1, 2048 / L) * default_bs),
        "DeepRV + ScanTransfomer": int(min(1, 512 / L) * default_bs),
        "DeepRV + DKA": int(min(1, 1024 / L) * default_bs),
    }[model_name]
    train_steps = default_steps * (default_bs // bs)
    if model_name == "Inducing DeepRV + gMLP":
        train_steps *= 2
    optimizer = optax.yogi(cosine_annealing_lr(train_steps, max_lr))
    if model_name in ["DeepRV + ScanTransfomer", "DeepRV + DKA"]:
        optimizer = optax.adamw(max_lr)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optimizer)
    return optimizer, max_lr, bs, train_steps


def plot_model_scalability_metrics(result_df: pd.DataFrame, save_dir: Path):
    """
    Plots scalability and performance metrics across grid sizes for multiple models.
    Assumes 'grid_size' and 'model_name' columns exist in `result_df`.
    """
    sns.set_theme(style="whitegrid")
    models = result_df["model_name"].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = {model: palette[i] for i, model in enumerate(models)}
    result_df.infer_flops = result_df.infer_flops / result_df.bs
    result_df.train_flops = result_df.train_flops / result_df.bs

    def _plot_metric_group(
        ax, metric_cols: Union[str, List[str]], title: str, ylabel: str
    ):
        if isinstance(metric_cols, str):
            metric_cols = [metric_cols]
        for model in models:
            df_m = result_df[result_df["model_name"] == model]
            color = model_colors[model]
            for metric in metric_cols:
                if metric not in df_m.columns or df_m[metric].isnull().all():
                    continue
                label = f"{model}" if len(metric_cols) == 1 else f"{model} - {metric}"
                ax.plot(
                    df_m["grid_size"],
                    df_m[metric],
                    label=label,
                    marker="o",
                    color=color,
                )
        ax.set_title(title)
        ax.set_xlabel("Grid Size")
        ax.set_ylabel(ylabel)
        ax.legend()

    # NOTE: time
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(axes[0], "train_time", "Training Time", "Seconds")
    _plot_metric_group(axes[1], "infer_time", "Inference Time", "Seconds")
    _plot_metric_group(axes[2], "total_time", "Total Time", "Seconds")
    plt.tight_layout()
    plt.savefig(save_dir / "speed.png")
    # NOTE: scalability
    _, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    _plot_metric_group(axes[0], "parameters", "Parameter Count", "Count")
    _plot_metric_group(
        axes[1], ["infer_flops", "train_flops"], "GFLOPs per sample", "GFLOPs"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "flops.png")
    # NOTE: performance
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(axes[0], "MSE(y, y_hat)", "Prediction MSE", "MSE")
    _plot_metric_group(
        axes[1],
        "ls wasserstein distance",
        "Lengthscale Wasserstein Distance",
        "Distance",
    )
    _plot_metric_group(axes[2], "ESS ls", "Lengthscale ESS", "ESS")
    plt.tight_layout()
    plt.savefig(save_dir / "performance.png")


class LogScaleTransform(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive
    event_dim = 0  # Scalar transform

    def __call__(self, x):
        return jnp.exp(x * jnp.log(100.0))

    def _inverse(self, y):
        return jnp.log(y) / jnp.log(100.0)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(100.0) + x * jnp.log(100.0)


# def load_surr_model_results(model, model_name, grid_size, optimizer, ls=10):
#     from orbax.checkpoint import PyTreeCheckpointer

#     grid_path = Path(
#         f"results/scalability_log_priors_ls_{ls}/grid_{grid_size}"
#     ).absolute()
#     model_path = grid_path / f"{model_name}"
#     ckptr = PyTreeCheckpointer()
#     ckpt = ckptr.restore(model_path / "model.ckpt")
#     state = TrainState.create(
#         apply_fn=model.apply,
#         tx=optimizer,
#         params=ckpt["state"]["params"],
#         kwargs=ckpt["state"]["kwargs"],
#     )
#     surrogate_decoder = generate_surrogate_decoder(state, model)
#     res_df = pd.read_csv(grid_path / "res.csv")
#     res_df = res_df[res_df["model_name"] == model_name]
#     train_time, eval_mse = (
#         res_df["train_time"].values[0],
#         res_df["Test Norm MSE"].values[0],
#     )
#     infer_flops, train_flops = (
#         res_df["infer_flops"].values[0],
#         res_df["train_flops"].values[0],
#     )
#     parameters = res_df["parameters"].values[0]
#     return train_time, eval_mse, surrogate_decoder, infer_flops, train_flops, parameters


# def load_inference_results(model_name, grid_size, ls=10.0):
#     res_dir = Path(
#         f"results/scalability_log_priors_ls_{ls}/grid_{grid_size}/{model_name}/"
#     )
#     with open(res_dir / "hmc_pp.pkl", "rb") as load_f:
#         post = pickle.load(load_f)
#     with open(res_dir / "hmc_samples.pkl", "rb") as load_f:
#         samples = pickle.load(load_f)
#     return samples, post


# def load_single_res(model_name, grid_size, ls=10.0):
#     res_dir = Path(
#         f"results/scalability_log_priors_ls_{ls}/grid_{grid_size}/{model_name}/single_res.pkl"
#     )
#     with open(res_dir, "rb") as load_f:
#         single_res = pickle.load(load_f)
#     if isinstance(single_res["parameters"], str):
#         single_res["parameters"] = int(
#             single_res["parameters"]
#             .split("Total Parameters: ")[-1]
#             .split(" ")[0]
#             .replace(",", "")
#         )
#     return single_res


if __name__ == "__main__":
    main()
