import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import flax.linen as nn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random, vmap
from jax.scipy.linalg import solve_triangular
from numpyro import deterministic, sample
from numpyro import distributions as dist
from numpyro.distributions.transforms import ParameterFreeTransform
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from sps.kernels import matern_1_2
from sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    generate_surrogate_decoder,
    inducing_deep_rv_train_step,
)


def main(seed=55, logged_priors=False, max_ls=100.0):
    wandb.init(mode="disabled")  # NOTE: downstream function assumes active wandb
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path(f"results/inducing_drv{'_log_priors' if logged_priors else ''}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    grid_dim = 32
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": grid_dim}] * 2).reshape(-1, 2)
    models = {
        "DeepRV inv simple + gMLP": (gMLPDeepRV(num_blks=2), deep_rv_train_step),
        "DeepRV inv + gMLP": (gMLPDeepRV(num_blks=2), inducing_deep_rv_train_step),
        "DeepRV + gMLP": (gMLPDeepRV(num_blks=2), deep_rv_train_step),
        "Inducing Points": (None, None),
    }
    y_obs = gen_y_obs(rng_obs, s)
    obs_mask = generate_obs_mask(rng_idxs, y_obs, 0.3)
    log_ls = dist.TransformedDistribution(
        dist.Beta(4.0, 1.0), LogScaleTransform(max_ls=max_ls)
    )
    priors = {
        "ls": log_ls if logged_priors else dist.Uniform(1.0, max_ls),
        "beta": dist.Normal(),
    }
    y_hats, all_samples, result = [], [], []
    L_train = int(s.shape[0] ** 0.75)

    for model_name, (nn_model, train_step) in models.items():
        solve_inv = "inv" in model_name
        infer_model, cond_names, s_train = inference_model_inducing_points(
            s, priors, obs_mask, L_train, solve_inv
        )
        train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
        eval_f_mse = None
        if nn_model is not None:
            loader = gen_train_dataloader(s, s_train, priors, solve_inv, batch_size=32)
            (save_dir / f"{model_name}").mkdir(parents=True, exist_ok=True)
            train_time, eval_mse, surrogate_decoder, eval_f_mse = surrogate_model_train(
                rng_train,
                rng_test,
                loader,
                nn_model,
                train_step,
                solve_inv,
                save_dir / f"{model_name}",
            )
        samples, mcmc, post, infer_time = hmc(
            rng_infer, infer_model, y_obs, obs_mask, surrogate_decoder
        )
        ess = az.ess(mcmc, method="mean")
        plot_infer_trace(
            samples,
            mcmc,
            None,
            cond_names,
            save_dir / f"{model_name}_infer_trace.png",
        )

        y_hats.append(post["obs"])
        all_samples.append(samples)
        result.append(
            {
                "model_name": model_name,
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "Test f MSE": eval_f_mse,
                "infer_time": infer_time,
                "inferred lengthscale mean": samples["ls"].mean(axis=0),
                "inferred fixed effects": samples["beta"].mean(axis=0),
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "ESS spatial effects": ess["f"].mean().item() if ess else None,
                "ESS lengthscale": ess["ls"].item() if ess else None,
                "ESS fixed effects": ess["beta"].item() if ess else None,
            }
        )
    model_names = list(models.keys())
    try:
        plot_posterior_predictive_comparisons(
            all_samples,
            {},
            priors,
            model_names,
            cond_names,
            save_dir / "comp",
            "Inducing Points",
        )
    except Exception:
        pass
    plot_models_predictive_means(
        s,
        s_train,
        grid_dim,
        y_obs,
        y_hats,
        obs_mask,
        model_names,
        save_dir / "obs_means.png",
    )
    wass_dist = posterior_wasserstein_distance(all_samples, model_names, cond_names)
    result = pd.DataFrame(result)
    for model_name in model_names:
        if model_name == "Inducing Points":
            continue
        for n in cond_names:
            result.loc[
                result["model_name"] == model_name, f"{n} wasserstein distance"
            ] = wass_dist[model_name][n]
    result.to_csv(save_dir / "res.csv")
    # from diagnostics import compare_grads, diff_per_loader

    # _, _, s_train = inference_model_inducing_points(s, priors, obs_mask, L_train, False)
    # diff_per_loader(models, s, s_train, matern_1_2, save_dir, max_ls)
    # compare_grads(
    #     models,
    #     s,
    #     inference_model_inducing_points,
    #     priors,
    #     obs_mask,
    #     L_train,
    #     save_dir,
    #     target="f",
    #     max_ls=max_ls,
    # )
    # compare_grads(
    #     models,
    #     s,
    #     inference_model_inducing_points,
    #     priors,
    #     obs_mask,
    #     L_train,
    #     save_dir,
    #     target="f_u_bar",
    #     max_ls=max_ls,
    # )
    # exit(0)


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=50))
    k1, k2 = random.split(rng)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=1_000, num_warmup=4_00)
    mcmc = MCMC(nuts, num_chains=2, num_samples=5_000, num_warmup=2_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2)
    return samples, mcmc, post, infer_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    train_step,
    solve_inv: bool,
    results_dir: Path,
    train_num_steps: int = 400_000,
    valid_interval: int = 100_000,
    valid_steps: int = 5_000,
):
    valid_step = partial(inducing_valid_step, solve_inv=solve_inv)
    lr_schedule = cosine_annealing_lr(train_num_steps, 5.0e-3)
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
        valid_monitor_metric="f MSE",
    )
    train_time = (datetime.now() - start).total_seconds()
    metrics = evaluate(rng_test, state, valid_step, loader, valid_steps)
    eval_mse, eval_f_mse = metrics["norm MSE"], metrics["f MSE"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    with open(results_dir / "train_time.pkl", "wb") as out_file:
        pickle.dump(
            {"train_time": train_time, "eval_mse": eval_mse, "eval f mse": eval_f_mse},
            out_file,
        )
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder, eval_f_mse


@jit
def jit_solve_func(M, v):
    return vmap(jnp.linalg.solve, in_axes=(None, 0))(M, v)


@partial(jit, static_argnames=["lower"])
def jit_trin_solve_func(L_T, z, lower=False):
    s_trin = partial(solve_triangular, lower=lower)
    return vmap(s_trin, in_axes=(None, 0))(L_T, z)


def gen_train_dataloader(
    s: Array, s_train: Array, priors: dict, solve_inv: bool, batch_size: int
):
    var = 1.0
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    kernel_jit = jit(lambda s1, s2, var, ls: matern_1_2(s1, s2, var, ls))
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s_train.shape[0]))
            K = kernel_jit(s_train, s_train, var, ls) + jitter
            L = jnp.linalg.cholesky(K)
            f = f_jit(L, z)
            if solve_inv:
                f = jit_trin_solve_func(L.T, z, False)
            K_su = kernel_jit(s, s_train, var, ls)

            yield {
                "s": s_train,
                "f": f,
                "z": z,
                "conditionals": jnp.array([ls]),
                "L_uu": L,
                "K_su": K_su,
            }

    return dataloader


def inference_model_inducing_points(
    s: Array, priors: dict, obs_mask: Array, num_points: int, solve_inv: bool
):
    """Builds a poisson likelihood inference model for inducing points"""
    kmeans = KMeans(n_clusters=num_points, random_state=0)
    u = kmeans.fit(s[obs_mask]).cluster_centers_  # shape (num_points, s.shape[1])
    surr_kwargs = {"s": u}  # we train deepRV with u
    jitter = 5e-4 * jnp.eye(u.shape[0])

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = sample("ls", priors["ls"], sample_shape=())
        beta = sample("beta", priors["beta"], sample_shape=())
        K_su = matern_1_2(s, u, var, ls)
        z = sample("z", dist.Normal(), sample_shape=(1, u.shape[0]))
        if surrogate_decoder is not None and solve_inv:
            f_bar_u = surrogate_decoder(z, jnp.array([ls]), **surr_kwargs).squeeze()
            f_bar_u = deterministic("f_u_bar", f_bar_u)
        else:
            K_uu = matern_1_2(u, u, var, ls) + jitter
            L_uu = jnp.linalg.cholesky(K_uu)
            if surrogate_decoder is not None:
                f_u = surrogate_decoder(z, jnp.array([ls]), **surr_kwargs).squeeze()
                f_bar_u = solve_triangular(
                    L_uu.T, solve_triangular(L_uu, f_u, lower=True), lower=False
                )
                f_bar_u = deterministic("f_u_bar", f_bar_u)
            else:
                f_bar_u = deterministic(
                    "f_u_bar", solve_triangular(L_uu.T, z[0], lower=False)
                )
        f = deterministic("f", K_su @ f_bar_u)
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            return sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing, ["ls", "beta"], u


@partial(jit, static_argnames=["solve_inv"])
def inducing_valid_step(rng, state, batch, solve_inv):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    f_bar_u, L_uu = batch["f"], batch["L_uu"]
    f_bar_u_hat = output.f_hat
    if not solve_inv:
        f_bar_u = jit_trin_solve_func(L_uu.T, batch["z"], False)
        f_bar_u_hat = jit_trin_solve_func(
            L_uu.T, jit_trin_solve_func(L_uu, f_bar_u_hat, True), False
        )
    residuals = f_bar_u.squeeze() - f_bar_u_hat.squeeze()
    f_mse = (0.5 * (jnp.einsum("ij, bj-> bi", batch["K_su"], residuals)) ** 2).mean()
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"], "f MSE": f_mse}


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
    s,
    s_train,
    grid_dim,
    f_obs,
    f_hats,
    obs_mask,
    model_names,
    save_path: Path,
    log=True,
):
    f_hat_means = [f_mean.mean(axis=0).reshape(grid_dim, grid_dim) for f_mean in f_hats]
    f_obs = f_obs.reshape(grid_dim, grid_dim)
    if log:
        f_hat_means = [jnp.log(f + 1) for f in f_hat_means]
        f_obs = jnp.log(f_obs + 1)
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    vmax, vmin = max(vmax, f_obs.max().item()), min(vmax, f_obs.min().item())
    cols = 3
    rows = int(jnp.ceil((len(f_hat_means) + 3) / cols))
    fig, ax = plt.subplots(
        rows, cols, figsize=(6 * cols, 7 * rows), constrained_layout=True
    )
    ax = ax.flatten()
    distances_sq = jnp.sum((s[:, None, :] - s_train[None, :, :]) ** 2, axis=-1)
    closest_s_indices = jnp.argmin(distances_sq, axis=0)
    closest_mask = jnp.zeros(s.shape[0], dtype=bool)
    closest_mask = closest_mask.at[closest_s_indices].set(True)
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(grid_dim, grid_dim), f_obs)
    f_train = np.ma.masked_where(~closest_mask.reshape(grid_dim, grid_dim), f_obs)
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    ax[0].set_title("y observed")
    ax[1].imshow(f_train, origin="lower", cmap=cmap)
    ax[1].set_title("y train")
    ax[2].imshow(f_obs, vmin=vmin, vmax=vmax, origin="lower")
    ax[2].set_title("y")
    for i, f_mean in enumerate(f_hat_means, start=3):
        model_name = model_names[i - 3]
        im = ax[i].imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower")
        ax[i].set_title("Mean " r"$\hat{y}$" f" {model_name}")
    for i in range(len(ax)):
        ax[i].set_axis_off()
        if (i + 1) % cols == 0:
            fig.colorbar(im, ax=ax[i])
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


class LogScaleTransform(ParameterFreeTransform):
    domain = dist.constraints.real
    codomain = dist.constraints.positive
    event_dim = 0  # Scalar transform

    def __init__(self, max_ls=100.0):
        self.max_ls = max_ls

    def __call__(self, x):
        return jnp.exp(x * jnp.log(self.max_ls))

    def _inverse(self, y):
        return jnp.log(y) / jnp.log(self.max_ls)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return jnp.log(self.max_ls) + x * jnp.log(self.max_ls)


def posterior_wasserstein_distance(
    samples: list, model_names: list[str], var_names: list[str]
):
    """
    Computes Wasserstein distance for each variable between the posterior distributions
    of each model and the baseline "Baseline_GP".
    """
    baseline_index = model_names.index("Inducing Points")
    baseline_samples = samples[baseline_index]
    distances = {m: {} for m in model_names if m != "Inducing Points"}
    for model_name, model_sample in zip(model_names, samples):
        if model_name == "Inducing Points":
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


if __name__ == "__main__":
    main()
