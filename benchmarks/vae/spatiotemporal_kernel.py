import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
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
import seaborn as sns
from deep_rv_paper import display_hyperparam
from jax import Array, jit, random
from jax.scipy.linalg import solve_triangular
from numpyro import distributions as dist
from numpyro.distributions.transforms import ParameterFreeTransform
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam
from omegaconf import DictConfig
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from dl4bi_sps.utils import build_grid
from utils.plot_utils import plot_infer_trace, plot_posterior_predictive_comparisons

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import PriorCVAE, gMLPDeepRV
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    generate_surrogate_decoder,
)


def main(seed=19, time_steps=5, grid_shape=(16, 16)):
    wandb.init(mode="disabled")  # NOTE: downstream function assumes active wandb
    t = jnp.arange(time_steps, dtype=jnp.float32)
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path("results/spatiotemporal/")
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": grid_shape[0]}] * 2).reshape(
        -1, 2
    )
    L, D = s.shape[0], s.shape[1]
    T = time_steps
    models = {
        "Baseline_GP": None,
        "Inducing Points Large": None,
        "PriorCVAE": PriorCVAE(
            MLP(dims=[T * L, T * L]), MLP(dims=[T * L, T * L]), cond_as_locs, T * L
        ),
        "ADVI": None,
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
    }
    y_obs = gen_y_obs(rng_obs, s, t)
    spat_obs_mask = gen_spatial_obs_mask(rng_idxs, grid_shape, obs_ratio=0.5)
    t_obs_mask = jnp.array([True, False, False, True, True])
    obs_mask = jnp.full((time_steps, s.shape[0]), False, dtype=jnp.bool)
    obs_mask = obs_mask.at[t_obs_mask].set(spat_obs_mask)
    log_ls = dist.TransformedDistribution(dist.Beta(4.0, 1.0), LogScaleTransform())
    priors = {
        "ls": log_ls,
        "a": dist.LogNormal(0.0, 1.0),
        "alpha": dist.Beta(2.0, 2.0),
        "nu": dist.Uniform(D, 2 * D),
        "beta": dist.Normal(),
    }
    poisson_inducing_llk = inference_model_inducing_points(
        s, t, priors, spat_obs_mask, t_obs_mask, num_spatial_pts=32, num_t_pts=2
    )
    poisson_inducing_llk_large = inference_model_inducing_points(
        s, t, priors, spat_obs_mask, t_obs_mask, num_spatial_pts=64, num_t_pts=2
    )
    loader = gen_train_dataloader(s, t, priors, batch_size=16)
    y_hats, all_samples, result = [], [], []
    for model_name, nn_model in models.items():
        poisson_llk, cond_names = inference_model(s, t, priors, model_name)
        model_path = save_dir / f"{model_name}"
        model_path.mkdir(parents=True, exist_ok=True)
        infer_model = {
            "Inducing Points": poisson_inducing_llk,
            "Inducing Points Large": poisson_inducing_llk_large,
        }.get(model_name, poisson_llk)
        train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
        if nn_model is not None:
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train, rng_test, loader, model_name, nn_model, model_path
            )
        if model_name != "ADVI":
            samples, mcmc, post, infer_time = hmc(
                rng_infer, infer_model, y_obs, obs_mask, model_path, surrogate_decoder
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
        sq_res = (y_obs - post["obs"].mean(axis=0)) ** 2
        res = {
            "model_name": model_name,
            "train_time": train_time,
            "Test Norm MSE": eval_mse,
            "infer_time": infer_time,
            "MSE(y, y_hat)": (sq_res).mean(),
            "obs MSE(y, y_hat)": (sq_res[obs_mask]).mean(),
            "unobs MSE(y, y_hat)": (sq_res[jnp.logical_not(obs_mask)]).mean(),
        }
        res.update({f"inferred {c} mean": samples[c].mean(axis=0) for c in cond_names})
        res.update(
            {
                f"ESS {c}": ess[c].mean().item() if ess else None
                for c in cond_names + ["mu"]
            }
        )
        result.append(res)
    model_names = list(models.keys())
    plot_posterior_predictive_comparisons(
        all_samples, {}, priors, model_names, cond_names, save_dir / "comp"
    )
    plot_models_predictive_means(
        y_obs,
        y_hats,
        obs_mask,
        model_names,
        time_steps,
        save_dir / "obs_means.png",
        grid_shape,
    )
    plot_models_predictive_means(
        y_obs,
        [
            ys
            for i, ys in enumerate(y_hats)
            if model_names[i] not in ["ADVI", "PriorCVAE"]
        ],
        obs_mask,
        [m for m in model_names if m not in ["ADVI", "PriorCVAE"]],
        time_steps,
        save_dir / "top_obs_means.png",
        grid_shape,
    )
    plot_posterior_predictive_comparisons_kde(
        all_samples, model_names, cond_names, save_dir / "comp"
    )
    result = posterior_wasserstein_distance(
        result, all_samples, model_names, cond_names
    )
    result = posterior_mean_gp_dist(result, y_hats, model_names)
    result = pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    model_path: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=1_000, num_warmup=4_00)
    mcmc = MCMC(nuts, num_chains=4, num_samples=10_000, num_warmup=4_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
    post["infer_time"] = infer_time
    with open(model_path / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(model_path / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    return samples, mcmc, post, infer_time


def advi(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Array,
    model_path: Path,
    num_steps: int = 50_000,
):
    rng_advi, rng_pp, rng_post = random.split(rng, 3)
    guide = AutoMultivariateNormal(model)
    optimizer = Adam(step_size=0.0001)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    start = datetime.now()
    svi_result = svi.run(
        rng_advi, num_steps, y=y_obs, obs_mask=obs_mask, stable_update=True
    )
    infer_time = (datetime.now() - start).total_seconds()
    params = svi_result.params
    samples = guide.sample_posterior(rng_pp, params, sample_shape=(40_000,))
    post = Predictive(model, samples)(rng_post)
    post["infer_time"] = infer_time
    with open(model_path / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(model_path / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    return samples, post, infer_time


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model_name: str,
    model: nn.Module,
    model_path: Path,
    train_num_steps: int = 500_000,
    valid_interval: int = 100_000,
    valid_steps: int = 5_000,
):
    lr_schedule = cosine_annealing_lr(train_num_steps, 5.0e-3)
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
    save_ckpt(state, DictConfig({}), model_path / "model.ckpt")
    with open(model_path / "train_time.pkl", "wb") as out_file:
        pickle.dump({"train_time": train_time, "eval_mse": eval_mse}, out_file)
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(s: Array, t: Array, priors: dict, batch_size=32):
    T, L, D = t.shape[0], s.shape[0], s.shape[1]
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))
    s_expanded = jnp.broadcast_to(s, (T, L, D))
    t_expanded = t[:, None, None] * jnp.ones((1, L, 1))
    st = jnp.concatenate([s_expanded, t_expanded], axis=-1).reshape(T * L, D + 1)
    jitter = 5e-4 * jnp.eye(st.shape[0])

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_a, rng_alpha, rng_nu, rng_z = random.split(
                rng_data, 6
            )
            var, b = 1.0, 1.0
            ls = priors["ls"].sample(rng_ls)
            a = priors["a"].sample(rng_a)
            alpha = priors["alpha"].sample(rng_alpha)
            nu = priors["nu"].sample(rng_nu)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, st.shape[0]))
            K = gneiting_covariance_matrix(s, t, s, t, var, ls, a, alpha, b, nu)
            K += jitter
            f = f_jit(K, z)
            yield {
                "s": st,
                "f": f,
                "z": z,
                "K": K,
                "conditionals": jnp.array([ls, a, alpha, nu]),
            }

    return dataloader


def inference_model(s: Array, t: Array, priors: dict, model_name):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    T, L, D = t.shape[0], s.shape[0], s.shape[1]
    s_expanded = jnp.broadcast_to(s, (T, L, D))
    t_expanded = t[:, None, None] * jnp.ones((1, L, 1))
    st = jnp.concatenate([s_expanded, t_expanded], axis=-1).reshape(T * L, D + 1)
    surrogate_kwargs = {"s": st}
    if "kernelAttn" in model_name:
        surrogate_kwargs["K"] = None

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var, b = 1.0, 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        a = numpyro.sample("a", priors["a"], sample_shape=())
        alpha = numpyro.sample("alpha", priors["alpha"], sample_shape=())
        nu = numpyro.sample("nu", priors["nu"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, T * L))
        if "K" in surrogate_kwargs or surrogate_decoder is None:
            K = gneiting_covariance_matrix(s, t, s, t, var, ls, a, alpha, b, nu)
            K += 5e-4 * jnp.eye(T * L)
            surrogate_kwargs["K"] = K
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(
                    z, jnp.array([ls, a, alpha, nu]), **surrogate_kwargs
                ).reshape(T, L),
            )
        else:
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0])).reshape(T, L)
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson, ["ls", "a", "alpha", "nu", "beta"]


def inference_model_inducing_points(
    s: Array,
    t: Array,
    priors: dict,
    spat_obs_mask: Array,
    t_obs_mask: Array,
    num_spatial_pts: int,
    num_t_pts: int,
):
    """Builds a poisson likelihood inference model for inducing points"""
    T, L = t.shape[0], s.shape[0]
    kmeans_space = KMeans(n_clusters=num_spatial_pts, random_state=0)
    s_u = kmeans_space.fit(s[spat_obs_mask]).cluster_centers_  # (S_U, D)
    kmeans_time = KMeans(n_clusters=num_t_pts, random_state=1)
    t_u = kmeans_time.fit(t[t_obs_mask, None]).cluster_centers_.flatten()  # (T_U,)
    D_U = s_u.shape[0] * t_u.shape[0]

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var, b = 1.0, 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        a = numpyro.sample("a", priors["a"], sample_shape=())
        alpha = numpyro.sample("alpha", priors["alpha"], sample_shape=())
        nu = numpyro.sample("nu", priors["nu"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_uu = gneiting_covariance_matrix(s_u, t_u, s_u, t_u, var, ls, a, alpha, b, nu)
        K_uu += 5e-4 * jnp.eye(D_U)
        L_uu = jnp.linalg.cholesky(K_uu)
        K_su = gneiting_covariance_matrix(s, t, s_u, t_u, var, ls, a, alpha, b, nu)
        z = numpyro.sample("z", dist.Normal(), sample_shape=(D_U,))
        f = K_su @ solve_triangular(L_uu.T, z, lower=False)
        f = numpyro.deterministic("mu", f.reshape(T, L))
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_y_obs(rng: Array, s: Array, t: Array):
    """generates a poisson observed data sample for inference"""
    T, L = t.shape[0], s.shape[0]
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, 20.0, 1.0
    a, alpha, b, nu = 0.5, 0.8, 1.0, 1.0
    K = gneiting_covariance_matrix(s, t, s, t, var, ls, a, alpha, b, nu)
    K = K + 5e-4 * jnp.eye(T * L)
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu).reshape(T, L)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def gen_spatial_obs_mask(rng: Array, grid_shape: tuple, obs_ratio: float = 0.15):
    """
    Generates a spatial observation mask for a 2D grid. Keeps a certain percentage of the domain unmasked,
    in the form of a few spatially-contiguous elliptical blobs. The output is a 1D boolean mask indicating
    which locations are observed.

    Args:
        rng: JAX PRNG key
        y_obs: Flattened signal (L,)
        grid_shape: Tuple (H, W) for reshaping the 1D signal
        obs_ratio: Fraction of the total grid to remain observed

    Returns:
        mask_flat: Flattened boolean mask of shape (L,), where True = observed, False = masked
    """
    H, W = grid_shape
    total_points = H * W
    num_obs_points = int(obs_ratio * total_points)
    mask = jnp.zeros((H, W), dtype=bool)

    points_collected = 0
    blob_idx = 0
    while points_collected < num_obs_points:
        rng_blob, rng = random.split(rng)
        rngs = random.split(rng_blob, 4)
        center_x = random.randint(rngs[0], (), 0, H)
        center_y = random.randint(rngs[1], (), 0, W)
        radius_x = random.randint(rngs[2], (), H // 8, H // 4)
        radius_y = random.randint(rngs[3], (), W // 8, W // 4)

        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        ellipse = (
            ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2
        ) <= 1.0
        new_mask = jnp.logical_or(mask, ellipse)
        added = jnp.sum(new_mask) - jnp.sum(mask)
        mask = new_mask
        points_collected += int(added)
        blob_idx += 1

    # NOTE: If we overshot, randomly drop extras
    if points_collected > num_obs_points:
        flat_idxs = jnp.argwhere(mask.flatten()).squeeze()
        rng_trim, _ = random.split(rngs[-1])
        selected = random.choice(
            rng_trim, flat_idxs, shape=(num_obs_points,), replace=False
        )
        final_mask = jnp.zeros(total_points, dtype=bool).at[selected].set(True)
    else:
        final_mask = mask.flatten()

    return final_mask


def plot_models_predictive_means(
    f_obs: Array,
    f_hats: list[Array],
    obs_mask: Array,
    model_names: list[str],
    time_steps: int,
    save_path: Path,
    grid_shape: tuple[int, int],
    log: bool = True,
):
    """
    Plots predictive means over time for multiple models, along with true and masked observations.

    Args:
        f_obs: (T, L) true latent field over time
        f_hats: list of (S, T, L) predictive samples
        obs_mask: (T, L) observation mask
        model_names: list of model names (len = len(f_hats))
        time_steps: number of time steps (T)
        save_path: path to save figure
        grid_shape: spatial grid shape (H, W) such that H * W == L
        log: whether to apply log transform to the values
    """
    T, L = f_obs.shape
    H, W = grid_shape
    assert H * W == L, "grid_shape must match number of spatial locations"
    # Compute predictive means (mean over samples)
    f_hat_means = [f.mean(axis=0).reshape(T, H, W) for f in f_hats]
    f_obs_grid = f_obs.reshape(T, H, W)
    obs_mask_grid = obs_mask.reshape(T, H, W)
    if log:
        f_obs_grid = jnp.log1p(f_obs_grid)
        f_hat_means = [jnp.log1p(f) for f in f_hat_means]
    # Determine color scale
    all_vals = [f_obs_grid] + f_hat_means
    vmin = min([f.min() for f in all_vals])
    vmax = max([f.max() for f in all_vals])
    n_models = len(model_names)
    n_rows = 2 + n_models  # 1 for y, 1 for y masked, rest for models
    fig, ax = plt.subplots(
        n_rows,
        time_steps,
        figsize=(4 * time_steps, 4 * n_rows),
        constrained_layout=True,
    )
    for t in range(time_steps):
        # True observation
        ax[0, t].imshow(f_obs_grid[t], vmin=vmin, vmax=vmax, origin="lower")
        ax[0, t].set_title(f"True $y$, t={t}")
        ax[0, t].set_axis_off()
        # Masked observation
        masked = np.ma.masked_where(~obs_mask_grid[t], f_obs_grid[t])
        cmap = plt.cm.viridis
        cmap.set_bad("black")
        ax[1, t].imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        ax[1, t].set_title(f"Observed $y$, t={t}")
        ax[1, t].set_axis_off()
        # Model predictive means
        for i, f_mean in enumerate(f_hat_means):
            im = ax[2 + i, t].imshow(f_mean[t], vmin=vmin, vmax=vmax, origin="lower")
            ax[2 + i, t].set_title(f"{model_names[i]}, t={t}")
            ax[2 + i, t].set_axis_off()
    for r in range(n_rows):
        fig.colorbar(im, ax=ax[r, -1], fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


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


@jit
def gneiting_covariance_matrix(
    s1: Array,
    t1: Array,
    s2: Array,
    t2: Array,
    var: float,
    ls: float,
    a: float,
    alpha: float,
    b: float,
    nu: float,
):
    """
    Gneiting (2002) nonseparable space-time kernel for Euclidean R^d x R.
    s: (L, d) spatial locations
    t: (T,) time points
    var: marginal process variance
    ls: spatial lengthscale
    a, alpha: temporal scaling and smoothness
    b: controls degree of nonseparability (0 = separable)
    nu: controls marginal temporal decay
    Returns:
      Covariance matrix of shape (T*L, T*L)
    """
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    h2 = jnp.sum((s1[:, None, :] - s2[None, :, :]) ** 2, axis=-1)  # (L, L)
    u = jnp.abs(t1[:, None] - t2[None, :])  # (T1, T2)
    h2_exp = h2[None, None, :, :]  # (1, 1, L1, L2)
    u_exp = u[:, :, None, None]  # (T1, T2, 1, 1)
    g = 1.0 + a * u_exp ** (2 * alpha)  # (T1, T2, 1, 1)
    # kernel: shape (T1, T2, L1, L2)
    K = var / (g**nu) * jnp.exp(-h2_exp / (ls**2 * g**b))
    K = K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)
    return K


def posterior_wasserstein_distance(
    result: list[dict], samples: list, model_names: list[str], var_names: list[str]
):
    """
    Computes Wasserstein distance for each variable between the posterior distributions
    of each model and the baseline "Baseline_GP".
    """
    baseline_index = model_names.index("Baseline_GP")
    baseline_samples = samples[baseline_index]
    for model_res, model_sample in zip(result, samples):
        for var_name in var_names:
            model_res[f"{var_name} wasserstein distance"] = jnp.nan
            if model_res["model_name"] == "Baseline_GP":
                continue
            baseline_var_samples = baseline_samples.get(var_name)
            model_var_samples = model_sample.get(var_name)
            if baseline_var_samples is not None and model_var_samples is not None:
                dist = wasserstein_distance(baseline_var_samples, model_var_samples)
                model_res[f"{var_name} wasserstein distance"] = dist
    return result


def posterior_mean_gp_dist(result: list[dict], y_hats: list, model_names: list[str]):
    baseline_index = model_names.index("Baseline_GP")
    y_hat_gp = y_hats[baseline_index].mean(axis=0)
    for model_res, y_hat in zip(result, y_hats):
        if model_res["model_name"] == "Baseline_GP":
            model_res["MSE(y_hat_gp, y_hat)"] = jnp.nan
        else:
            model_res["MSE(y_hat_gp, y_hat)"] = jnp.mean(
                (y_hat_gp - y_hat.mean(axis=0)) ** 2
            )
    return result


def plot_posterior_predictive_comparisons_kde(
    samples: list[dict],
    model_names: list[str],
    var_names: list[str],
    save_prefix: Path,
):
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(n_vars * 1.7, 2.0), sharey=False)
    if n_vars == 1:
        axes = [axes]
    all_handles, all_labels = None, None
    for i, var_name in enumerate(var_names):
        ax = axes[i]
        min_val, max_val = np.inf, -np.inf
        for model_name, model_dict in zip(model_names, samples):
            model_samples = model_dict.get(str(var_name), None)
            if model_samples is not None:
                if var_name == "a":
                    model_samples = model_samples[model_samples <= 2]
                min_val = min(min_val, model_samples.min())
                max_val = max(max_val, model_samples.max())
                model_n = model_name.split(" +")[0]
                sns.kdeplot(
                    model_samples, label=model_n, linewidth=1.2, alpha=0.7, ax=ax
                )
        ax.set_xlabel(display_hyperparam(var_name), fontsize=9)
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)
        if i == 0:
            all_handles, all_labels = ax.get_legend_handles_labels()
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),  # closer to plots
            ncol=len(all_labels),
            fontsize=8,
            frameon=False,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave just enough space for legend
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{save_prefix}_posterior_kde.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
