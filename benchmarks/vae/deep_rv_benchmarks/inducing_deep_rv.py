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
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import FixedKernelAttention, gMLPDeepRV
from dl4bi.vae.train_utils import (
    deep_rv_train_step,
    generate_surrogate_decoder,
    inducing_deep_rv_train_step,
    inducing_deep_rv_train_step_latent_plus_residual,
)


def main(seed=55, logged_priors=False, max_ls=40.0, grid_dim=100, L_train=512):
    wandb.init(mode="disabled")  # NOTE: downstream function assumes active wandb
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path(
        f"results/inducing_drv{'_log_priors' if logged_priors else ''}"
        f"_{L_train}_{grid_dim**2}/"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": grid_dim}] * 2).reshape(-1, 2)
    models = {
        "Inducing Points": None,
        "DeepRV kernelAttn inv + gMLP latent": gMLPDeepRV(
            num_blks=4, attn=FixedKernelAttention(), head=MLP([128, 64, 1])
        ),
        "DeepRV kernelAttn inv + gMLP": gMLPDeepRV(
            num_blks=4, attn=FixedKernelAttention(), head=MLP([128, 64, 1])
        ),
        "DeepRV kernelAttn inv + gMLP simple": gMLPDeepRV(
            num_blks=4, attn=FixedKernelAttention(), head=MLP([128, 64, 1])
        ),
    }
    y_obs = gen_y_obs(rng_obs, s)
    obs_mask = generate_obs_mask(rng_idxs, y_obs, 0.5)
    log_ls = dist.TransformedDistribution(
        dist.Beta(4.0, 1.0), LogScaleTransform(max_ls=max_ls)
    )
    train_priors = {"ls": dist.Uniform(1.0, max_ls + 10.0)}
    infer_priors = {
        "ls": log_ls if logged_priors else dist.Uniform(1.0, max_ls),
        "beta": dist.Normal(),
    }
    y_hats, all_samples, result = [], [], []
    kmeans = KMeans(n_clusters=L_train, random_state=0)
    s_train = kmeans.fit(s[obs_mask]).cluster_centers_  # shape (num_points, s.shape[1])
    # old_res = pd.read_csv(save_dir / "res.csv")
    with open(save_dir / "s_train.pkl", "wb") as ff:
        pickle.dump(s_train, ff)
    for model_name, nn_model in models.items():
        model_dir = save_dir / f"{model_name}"
        # if (model_dir / "single_res.pkl").exists():
        #     with open(model_dir / "hmc_samples.pkl", "rb") as out_file:
        #         samples = pickle.load(out_file)
        #     with open(model_dir / "hmc_pp.pkl", "rb") as out_file:
        #         post = pickle.load(out_file)
        #     with open(model_dir / "single_res.pkl", "rb") as out_file:
        #         res = pickle.load(out_file)
        #     samples = {k: it for k, it in samples.items() if k in ["beta", "ls"]}
        #     y_hats.append(post["obs"])
        #     all_samples.append(samples)
        #     if len(old_res[old_res.model_name == model_name]) > 0:
        #         res = dict(old_res[old_res.model_name == model_name].iloc[0])
        #     result.append(res)
        #     continue
        model_dir.mkdir(parents=True, exist_ok=True)
        train_time, f_u_bar_mse, surrogate_decoder, ess = None, None, None, {}
        eval_f_mse, cosine_sim = None, None
        if nn_model is not None:
            loader = gen_train_dataloader(s, s_train, train_priors, batch_size=32)
            train_time, f_u_bar_mse, surrogate_decoder, eval_f_mse, cosine_sim = (
                surrogate_model_train(
                    rng_train, rng_test, loader, nn_model, model_dir, model_name
                )
            )
        infer_model, cond_names = inference_model_inducing_points(
            s, s_train, infer_priors, model_name
        )
        samples, mcmc, post, infer_time = hmc(
            rng_infer, infer_model, y_obs, obs_mask, model_dir, surrogate_decoder
        )
        ess = az.ess(mcmc, method="mean")
        try:
            plot_infer_trace(
                samples, mcmc, None, cond_names, model_dir / "infer_trace.png"
            )
        except Exception:
            pass

        y_hats.append(post["obs"])
        all_samples.append(samples)
        sq_res = (y_obs - post["obs"].mean(axis=0)) ** 2
        res = {
            "model_name": model_name,
            "train_time": train_time,
            "Test f_u_bar MSE": f_u_bar_mse,
            "Test f MSE": eval_f_mse,
            "Test loss cosine sim": cosine_sim,
            "infer_time": infer_time,
            "inferred lengthscale mean": samples["ls"].mean(axis=0),
            "inferred fixed effects": samples["beta"].mean(axis=0),
            "MSE(y, y_hat)": (sq_res).mean(),
            "obs MSE(y, y_hat)": (sq_res[obs_mask]).mean(),
            "unobs MSE(y, y_hat)": (sq_res[jnp.logical_not(obs_mask)]).mean(),
            "ESS spatial effects": ess["f"].mean().item() if ess else None,
            "ESS lengthscale": ess["ls"].item() if ess else None,
            "ESS fixed effects": ess["beta"].item() if ess else None,
        }
        with open(model_dir / "single_res.pkl", "wb") as ff:
            pickle.dump(res, ff)
        result.append(res)
    model_names = list(models.keys())
    try:
        plot_posterior_predictive_comparisons(
            all_samples,
            {},
            infer_priors,
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
    result = posterior_wasserstein_distance(
        result, all_samples, model_names, cond_names
    )
    result = posterior_mean_inducing_dist(result, y_hats, model_names)
    pd.DataFrame(result).to_csv(save_dir / "res.csv")
    # from inducing_drv_diagnostics import compare_grads, diff_per_loader

    # diff_per_loader(models, s, s_train, matern_1_2, save_dir, max_ls)
    # compare_grads(
    #     models,
    #     s,
    #     inference_model_inducing_points,
    #     infer_priors,
    #     s_train,
    #     L_train,
    #     save_dir,
    #     target="f",
    #     max_ls=max_ls,
    # )
    # compare_grads(
    #     models,
    #     s,
    #     inference_model_inducing_points,
    #     infer_priors,
    #     s_train,
    #     L_train,
    #     save_dir,
    #     target="f_u_bar",
    #     max_ls=max_ls,
    # )


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=50))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=2, num_samples=5_000, num_warmup=2_000)
    # mcmc = MCMC(nuts, num_chains=2, num_samples=10_000, num_warmup=4_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
    post["infer_time"] = infer_time
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    samples = {k: it for k, it in samples.items() if k in ["beta", "ls"]}
    return samples, mcmc, post, infer_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    results_dir: Path,
    model_name: str,
    train_num_steps: int = 800_000,
    valid_interval: int = 100_000,
    valid_steps: int = 5_000,
):
    lr_schedule = cosine_annealing_lr(train_num_steps, 1e-3)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.adamw(lr_schedule))
    train_step = {
        "DeepRV kernelAttn inv + gMLP simple": deep_rv_train_step,
        "DeepRV kernelAttn inv + gMLP": inducing_deep_rv_train_step,
        "DeepRV kernelAttn inv + gMLP latent": inducing_deep_rv_train_step_latent_plus_residual,
        "DeepRV kernelAttn inv + gMLP ksu loss": partial(
            inducing_deep_rv_train_step, weight=0.01
        ),
    }[model_name]
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        train_step,
        train_num_steps,
        loader,
        inducing_valid_step,
        valid_interval,
        valid_steps,
        loader,
        return_state="best",
        valid_monitor_metric="f MSE",
    )
    train_time = (datetime.now() - start).total_seconds()
    metrics = evaluate(rng_test, state, inducing_valid_step, loader, valid_steps)
    f_u_bar_mse, eval_f_mse = metrics["f_u_bar MSE"], metrics["f MSE"]
    cosine_similarity = metrics["loss cosine_sim"]
    save_ckpt(state, DictConfig({}), results_dir / "model.ckpt")
    with open(results_dir / "train_time.pkl", "wb") as out_file:
        pickle.dump(
            {
                "train_time": train_time,
                "eval f_u_bar mse": f_u_bar_mse,
                "eval f mse": eval_f_mse,
                "eval loss cosine_sim": cosine_similarity,
            },
            out_file,
        )
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, f_u_bar_mse, surrogate_decoder, eval_f_mse, cosine_similarity


def gen_train_dataloader(s: Array, s_train: Array, priors: dict, batch_size: int):
    var = 1.0
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    L_jit = jit(jnp.linalg.cholesky)
    s_trin = partial(solve_triangular, lower=False)
    jit_trin_solve_func = jit(vmap(s_trin, in_axes=(None, 0)))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s_train.shape[0]))
            K_uu = matern_1_2(s_train, s_train, var, ls) + jitter
            L_uu = L_jit(K_uu)
            f_u_bar = jit_trin_solve_func(L_uu.T, z)
            K_su = matern_1_2(s, s_train, var, ls)
            yield {
                "s": s_train,
                "f": f_u_bar,
                "z": z,
                "conditionals": jnp.array([ls]),
                "K_su": K_su,
                "K": K_uu,
            }

    return dataloader


def inference_model_inducing_points(s: Array, u: Array, priors: dict, model_name: str):
    """Builds a poisson likelihood inference model for inducing points"""
    surr_kwargs = {"s": u}  # we train deepRV with u
    if "kernelAttn" in model_name:
        surr_kwargs["K"] = None
    jitter = 5e-4 * jnp.eye(u.shape[0])

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = sample("ls", priors["ls"], sample_shape=())
        beta = sample("beta", priors["beta"], sample_shape=())
        K_su = matern_1_2(s, u, var, ls)
        z = sample("z", dist.Normal(), sample_shape=(1, u.shape[0]))
        if "K" in surr_kwargs or surrogate_decoder is None:
            K_uu = matern_1_2(u, u, var, ls) + jitter
            surr_kwargs["K"] = K_uu
        if surrogate_decoder is not None:
            f_bar_u = surrogate_decoder(z, jnp.array([ls]), **surr_kwargs).squeeze()
            f_bar_u = deterministic("f_u_bar", f_bar_u)
        else:
            L_uu = jnp.linalg.cholesky(K_uu)
            f_bar_u = deterministic(
                "f_u_bar", solve_triangular(L_uu.T, z[0], lower=False)
            )
        f = deterministic("f", K_su @ f_bar_u)
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            return sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing, ["ls", "beta"]


@jit
def inducing_valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    f_bar_u = batch["f"]
    f_bar_u_hat = output.f_hat
    residuals = f_bar_u.squeeze() - f_bar_u_hat.squeeze()
    f_bar_u_mse = 0.5 * (residuals**2).mean()
    proj_error = jnp.einsum("ij, bj-> bi", batch["K_su"], residuals)
    f_mse = 0.5 * (proj_error**2).mean()

    back_proj = jnp.einsum("ij, bj-> bi", batch["K_su"].T, proj_error)
    numerator = jnp.sum(back_proj * residuals, axis=1)
    denom = jnp.linalg.norm(back_proj, axis=1) * jnp.linalg.norm(residuals, axis=1)
    cosine_similarity = numerator / (denom + 1e-8)
    return {
        "f_u_bar MSE": f_bar_u_mse,
        "f MSE": f_mse,
        "loss cosine_sim": cosine_similarity.mean(),
    }


def gen_y_obs(rng: Array, s: Array):
    """generates a poisson observed data sample for inference"""
    rng_z, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, 10.0, 1.0
    K = matern_1_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    L = jnp.linalg.cholesky(K)
    z = dist.Normal().sample(rng_z, sample_shape=(s.shape[0],))
    mu = L @ z
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
    result: list[dict], samples: list, model_names: list[str], var_names: list[str]
):
    """
    Computes Wasserstein distance for each variable between the posterior distributions
    of each model and the baseline "Baseline_GP".
    """
    baseline_index = model_names.index("Inducing Points")
    baseline_samples = samples[baseline_index]
    for model_res, model_sample in zip(result, samples):
        for var_name in var_names:
            model_res[f"{var_name} wasserstein distance"] = jnp.nan
            if model_res["model_name"] == "Inducing Points":
                continue
            baseline_var_samples = baseline_samples.get(var_name)
            model_var_samples = model_sample.get(var_name)
            if baseline_var_samples is not None and model_var_samples is not None:
                dist = wasserstein_distance(baseline_var_samples, model_var_samples)
                model_res[f"{var_name} wasserstein distance"] = dist
    return result


def posterior_mean_inducing_dist(
    result: list[dict], y_hats: list, model_names: list[str]
):
    baseline_index = model_names.index("Inducing Points")
    y_hat_gp = y_hats[baseline_index].mean(axis=0)
    for model_res, y_hat in zip(result, y_hats):
        if model_res["model_name"] == "Inducing Points":
            model_res["MSE(y_hat_gp, y_hat)"] = jnp.nan
        else:
            model_res["MSE(y_hat_gp, y_hat)"] = jnp.mean(
                (y_hat_gp - y_hat.mean(axis=0)) ** 2
            )
    return result


# def load_surr_model_results(model, model_name, L, L_train):
#     from orbax.checkpoint import PyTreeCheckpointer

#     from dl4bi.core.train import TrainState

#     optimizer = optax.chain(
#         optax.clip_by_global_norm(3.0),
#         optax.adamw(cosine_annealing_lr(1_200_000, 1e-3), weight_decay=1e-2),
#     )
#     model_path = Path(f"results/inducing_drv_{L_train}_{L}/{model_name}").absolute()
#     ckptr = PyTreeCheckpointer()
#     ckpt = ckptr.restore(model_path / "model.ckpt")
#     print(model_path / "model.ckpt")
#     state = TrainState.create(
#         apply_fn=model.apply,
#         tx=optimizer,
#         params=ckpt["state"]["params"],
#         kwargs=ckpt["state"]["kwargs"],
#     )
#     surrogate_decoder = generate_surrogate_decoder(state, model)
#     with open(model_path / "train_time.pkl", "rb") as out_file:
#         dd = pickle.load(out_file)
#     train_time, f_u_bar_mse, eval_f_mse, cosine_similarity = (
#         dd["train_time"],
#         dd["eval f_u_bar mse"],
#         dd["eval f mse"],
#         dd["eval loss cosine_sim"],
#     )
#     return train_time, f_u_bar_mse, surrogate_decoder, eval_f_mse, cosine_similarity


if __name__ == "__main__":
    main()
