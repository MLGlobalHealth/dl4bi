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
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
from dl4bi_sps.kernels import matern_3_2
from dl4bi_sps.utils import build_grid
from utils.plot_utils import plot_infer_trace, plot_posterior_predictive_comparisons

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import (
    TrainState,
    cosine_annealing_lr,
    estimate_flops,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.vae import FixedKernelAttention, MLPDeepRV, PriorCVAE, gMLPDeepRV
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    generate_surrogate_decoder,
    prior_cvae_train_step,
)


def main(seed=42, logged_priors=True, gt_ls=10, grids=[16, 24, 32, 48, 64]):
    rng = random.key(seed)
    save_dir = Path(
        f"results/scalability{'_log_priors' if logged_priors else ''}_ls_{gt_ls}/"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    grids = [
        build_grid([{"start": 0.0, "stop": 100.0, "num": n}] * 2).reshape(-1, 2)
        for n in grids
    ]
    models = {
        "Baseline_GP": None,
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "DeepRV + gMLP kAttn": gMLPDeepRV(num_blks=2, attn=FixedKernelAttention()),
        "DeepRV + gMLP adamw": gMLPDeepRV(num_blks=2),
        "PriorCVAE": PriorCVAE,
        "DeepRV + MLP": MLPDeepRV,
        "Inducing Points": None,
        "Inducing Points Large": None,
        "ADVI": None,
        "RFF": None,
    }
    model_names = list(models.keys())
    log_ls = dist.TransformedDistribution(dist.Beta(4.0, 1.0), LogScaleTransform())
    priors = {
        "ls": log_ls if logged_priors else dist.Uniform(1.0, 100.0),
        "beta": dist.Normal(),
    }
    cond_names = list(priors.keys())
    for s in grids:
        result = []
        rng_train, rng_test, rng_infer, rng_idxs, rng_obs, rng = random.split(rng, 6)
        L, sqrt_L = s.shape[0], int(jnp.sqrt(s.shape[0]))
        if L == 1:  # NOTE: placeholder for fixing the rngs for gt_ls=50
            continue
        grid_s_path = save_dir / f"grid_{L}"
        grid_s_path.mkdir(parents=True, exist_ok=True)
        y_obs = gen_y_obs(rng_obs, s, gt_ls)
        obs_mask = gen_spatial_obs_mask(rng_idxs, (sqrt_L, sqrt_L), obs_ratio=0.5)
        u, u_large = get_inducing_points(s, obs_mask, L)
        gt_data = {"s": s, "u": u, "u_l": u_large, "obs_mask": obs_mask, "y_obs": y_obs}
        with open(grid_s_path / "GT_data.pkl", "wb") as ff:
            pickle.dump(gt_data, ff)
        y_hats, all_samples = [], []
        for model_name, nn_model in models.items():
            if "PriorCVAE" in model_name:
                nn_model = PriorCVAE(
                    MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L
                )
            elif model_name == "DeepRV + MLP":
                nn_model = MLPDeepRV(dims=[L, L])
            model_path = grid_s_path / f"{model_name}"
            model_path.mkdir(parents=True, exist_ok=True)
            is_inducing, kernel_attn = "Inducing" in model_name, "kAttn" in model_name
            infer_model = gen_infer_model(
                rng_infer,
                model_name,
                s,
                u,
                u_large,
                priors,
                L,
                is_inducing,
                kernel_attn,
            )
            train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
            infer_gflops, train_gflops, parameters = None, None, None
            max_lr, bs, num_train_steps = None, None, None
            if nn_model is not None:
                optimizer, max_lr, bs, num_train_steps, train_step = gen_train_params(
                    model_name, L
                )
                wandb.init(
                    config={
                        "model_name": model_name,
                        "grid_size": L,
                        "max_lr": max_lr,
                        "batch_size": bs,
                        "seed": seed,
                    },
                    mode="online",
                    name=f"{model_name}",
                    project="deep_rv_optimizations",
                    reinit=True,
                )
                loader = gen_train_dataloader(
                    s, u_large, priors, is_inducing, kernel_attn, batch_size=bs
                )
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
                    train_step,
                    nn_model,
                    model_path,
                    optimizer,
                    num_train_steps,
                )
                wandb.log({"train_time": train_time, "Test Norm MSE": eval_mse})
            if model_name != "ADVI":
                samples, mcmc, post, infer_time = hmc(
                    rng_infer,
                    infer_model,
                    y_obs,
                    obs_mask,
                    model_path,
                    L,
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
            sq_res = (y_obs - post["obs"].mean(axis=0)) ** 2
            res = {
                "model_name": model_name,
                "max_lr": max_lr,
                "bs": bs,
                "train_steps": num_train_steps,
                "grid_size": L,
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "infer_time": infer_time,
                "total_time": infer_time
                if train_time is None
                else infer_time + train_time,
                "MSE(y, y_hat)": (sq_res).mean(),
                "obs MSE(y, y_hat)": (sq_res[obs_mask]).mean(),
                "unobs MSE(y, y_hat)": (sq_res[jnp.logical_not(obs_mask)]).mean(),
                "infer_flops": infer_gflops,
                "train_flops": train_gflops,
                "parameters": parameters,
                "seed": seed,
                "num_chains": 4 if L == 32**2 else 1,
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
        result = posterior_wasserstein_distance(
            result, all_samples, model_names, cond_names
        )
        result = posterior_mean_gp_dist(result, y_hats, model_names)
        result = pd.DataFrame(result).to_csv(grid_s_path / "res.csv")
        plot_posterior_predictive_comparisons(
            all_samples, {}, priors, model_names, cond_names, grid_s_path / "comp"
        )
        plot_models_predictive_means(
            sqrt_L, y_obs, y_hats, obs_mask, model_names, grid_s_path / "obs_means.png"
        )
    aggregated_df = aggregate_csvs(save_dir)
    plot_model_scalability_metrics(aggregated_df, save_dir)


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    results_dir: Path,
    grid_size: int,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=1_000, num_warmup=4_00)
    num_chains = 4 if grid_size == 32**2 else 1
    mcmc = MCMC(nuts, num_chains=num_chains, num_samples=10_000, num_warmup=4_000)
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
    samples = {k: it for k, it in samples.items() if k in ["ls", "beta"]}
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
    train_step: Callable,
    model: nn.Module,
    results_dir: Path,
    optimizer,
    train_num_steps: int = 100_000,
    valid_interval: int = 25_000,
    valid_steps: int = 5_000,
):
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


def gen_train_dataloader(
    s: Array, u: Array, priors: dict, inducing=False, kernel_attn=False, batch_size=32
):
    s_train = u if inducing else s
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_3_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda L, z: jnp.einsum("ij,bj->bi", L, z))
    s_trin = partial(solve_triangular, lower=False)
    jit_trin_solve_func = jit(vmap(s_trin, in_axes=(None, 0)))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s_train.shape[0]))
            K = kernel_jit(s_train, var, ls)
            L = jnp.linalg.cholesky(K)
            batch = {"s": s_train, "z": z, "conditionals": jnp.array([ls])}
            if inducing:
                f = jit_trin_solve_func(L.T, z)
                K_su = matern_3_2(s, s_train, var, ls)
                batch.update({"f": f, "K_su": K_su})
            else:
                batch["f"] = f_jit(L, z)
            if kernel_attn:
                batch["K"] = K
            yield batch

    return dataloader


def gen_infer_model(
    rng_infer, model_name, s, u, u_large, priors, L, is_inducing, kernel_attn
):
    if model_name == "RFF":
        return inference_model_rff(rng_infer, s, priors, L * 2)
    if is_inducing:
        indc_pts = u
        if model_name in ["Inducing Points Large", "Inducing DeepRV + gMLP kAttn"]:
            indc_pts = u_large
        return inference_model_inducing_points(s, priors, indc_pts, kernel_attn)
    return inference_model(s, priors, kernel_attn)


def inference_model(s: Array, priors: dict, kernel_attn: bool):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if kernel_attn or surrogate_decoder is None:
            K = matern_3_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            surrogate_kwargs["K"] = K
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        else:
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson


def get_inducing_points(s, obs_mask, L):
    num_pts = int(min(int(2 * jnp.sqrt(L).item()), sum(obs_mask)))
    num_pts_large = int(min(int(2 * jnp.pow(L, 2 / 3).item()), sum(obs_mask)))
    kmeans = KMeans(n_clusters=num_pts, random_state=0)
    u = kmeans.fit(s[obs_mask]).cluster_centers_  # inducing points
    kmeans = KMeans(n_clusters=num_pts_large, random_state=0)
    u_large = kmeans.fit(s[obs_mask]).cluster_centers_  # inducing points large
    return u, u_large


def inference_model_inducing_points(s: Array, priors: dict, u: Array, kernel_attn):
    """Builds a poisson likelihood inference model for inducing points"""
    surr_kwargs = {"s": u}

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_su = matern_3_2(s, u, var, ls)
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, u.shape[0]))
        if kernel_attn or surrogate_decoder is None:
            K_uu = matern_3_2(u, u, var, ls) + 5e-4 * jnp.eye(u.shape[0])
            surr_kwargs["K"] = K_uu
        if surrogate_decoder is not None:
            f_u_bar = surrogate_decoder(z, jnp.array([ls]), **surr_kwargs).squeeze()
        else:
            L_uu = jnp.linalg.cholesky(K_uu)
            f_u_bar = solve_triangular(L_uu.T, z[0], lower=False)
        f = numpyro.deterministic("mu", K_su @ f_u_bar)
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing


def sample_matern32_omegas(key, M, d, ell=1.0):
    """Sample M base omegas for Matérn-3/2 with lengthscale `ell` (use ell=1.0 for base)."""
    nu = 1.5
    alpha = 2.0 * nu / (ell**2)  # = 3 / ell^2
    beta = nu + d / 2.0  # = (d+3)/2
    key_t, key_z = random.split(key)
    t = dist.Gamma(concentration=beta, rate=alpha).sample(key_t, (M,))
    z = random.normal(key_z, (M, d))
    omegas = z / jnp.sqrt(2.0 * t)[:, None]
    return omegas  # shape (M, d)


def inference_model_rff(rng, s, priors, feature_size=500):
    """
    Build a NumPyro Poisson-GP model using fixed RFF for Matérn-3/2.
    """
    M = feature_size
    # NOTE: sample RFF features once
    rng_omega, rng_phase = random.split(rng)
    base_omegas = sample_matern32_omegas(rng_omega, M, d=2, ell=1.0)
    phases = random.uniform(rng_phase, (M,), minval=0.0, maxval=2 * jnp.pi)
    base_proj = jnp.dot(s, base_omegas.T)
    jitter = 5e-4

    def poisson_rff(surrogate_decoder=None, obs_mask=True, y=None):
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        proj = base_proj / ls  # NOTE: = s @ (base_omegas / ls).T
        Phi = jnp.sqrt(2.0) * jnp.cos(proj + phases) / jnp.sqrt(M)
        z = numpyro.sample("z", dist.Normal(0.0, 1.0), sample_shape=(M,))
        mu_feat = jnp.dot(Phi, z)
        eps = numpyro.sample(
            "eps_jitter", dist.Normal(0.0, jnp.sqrt(jitter)), sample_shape=(s.shape[0],)
        )  # NOTE: the equivalent of adding jitter to GP
        mu = numpyro.deterministic("mu", mu_feat + eps)
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson_rff


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_y_obs(rng: Array, s: Array, gt_ls: float):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, gt_ls, 1.0
    K = matern_3_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
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
    grid_size, f_obs, f_hats, obs_mask, model_names, save_path: Path, log=True
):
    f_hat_means = [
        f_mean.mean(axis=0).reshape(grid_size, grid_size) for f_mean in f_hats
    ]
    f_obs = f_obs.reshape(grid_size, grid_size)
    if log:
        f_hat_means = [jnp.log(f + 1) for f in f_hat_means]
        f_obs = jnp.log(f_obs + 1)
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    cols = 4
    rows = int(jnp.ceil((len(f_hat_means) + 2) / cols))
    fig, ax = plt.subplots(
        rows, cols, figsize=(6 * cols, 7 * rows), constrained_layout=True
    )
    ax = ax.flatten()
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(grid_size, grid_size), f_obs)
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    ax[0].set_title("y observed")
    ax[1].imshow(f_obs, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
    ax[1].set_title("y")
    for i, f_mean in enumerate(f_hat_means, start=2):
        model_name = model_names[i - 2]
        im = ax[i].imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower", cmap=cmap)
        ax[i].set_title("Mean " r"$\hat{y}$" f" {model_name}")
    for i in range(len(ax)):
        ax[i].set_axis_off()
        if (i + 1) % cols == 0:
            fig.colorbar(im, ax=ax[i])
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


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


def gen_train_params(model_name, L, default_bs=32):
    default_steps = 300_000 if L >= 2048 else 200_000
    max_lr = {
        "Inducing DeepRV + gMLP kAttn": 1e-3,
        "DeepRV + gMLP": 5e-3 if L <= 32**2 else 1e-2,
        "PriorCVAE": 1.0e-3 if L <= 32**2 else 5e-3,
        "DeepRV + MLP": 1.0e-3 if L <= 32**2 else 5e-3,
        "DeepRV + gMLP kAttn": 1.0e-3 if L <= 32**2 else 2e-3,
        "DeepRV + gMLP adamw": 1.0e-3 if L <= 32**2 else 2e-3,
    }[model_name]
    bs = default_bs if L < 64**2 else default_bs // 2
    if model_name == "Inducing DeepRV + gMLP kAttn":
        bs = default_bs
    train_step = {"PriorCVAE": prior_cvae_train_step}.get(
        model_name, deep_rv_train_step
    )
    train_num_steps = default_steps * (default_bs // bs)
    if model_name == "Inducing DeepRV + gMLP kAttn":
        train_num_steps *= 2
    lr_schedule = cosine_annealing_lr(train_num_steps, max_lr)
    optimizer, clip = optax.yogi(lr_schedule), 3.0
    if "kAttn" in model_name or "adamw" in model_name:
        optimizer, clip = optax.adamw(lr_schedule, weight_decay=1e-2), 3.0
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optimizer)
    return optimizer, max_lr, bs, train_num_steps, train_step


def plot_model_scalability_metrics(
    result_df: pd.DataFrame, save_dir: Path, prefix: str = ""
):
    """
    Plots scalability and performance metrics across grid sizes for multiple models.
    Assumes 'grid_size' and 'model_name' columns exist in `result_df`.
    """
    sns.set_theme(style="whitegrid")
    models = result_df["model_name"].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = {model: palette[i] for i, model in enumerate(models)}

    # Normalize FLOPs per sample
    result_df["infer_time"] = result_df["infer_time"] / result_df["num_chains"]

    def _plot_metric_group(
        ax,
        metric_cols: Union[str, List[str]],
        title: str,
        ylabel: str,
        log_scale=False,
        loc="upper right",
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
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Grid Size", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if log_scale:
            ax.set_yscale("log")
        ax.legend(fontsize=8, loc=loc)
        ax.tick_params(axis="both", labelsize=9)

    # --- Time metrics with log scale
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(
        axes[0],
        "train_time",
        "Training Time",
        "Seconds",
        log_scale=True,
        loc="upper left",
    )
    _plot_metric_group(
        axes[1],
        "infer_time",
        "Inference Time",
        "Seconds",
        log_scale=True,
        loc="upper left",
    )
    _plot_metric_group(
        axes[2],
        "total_time",
        "Total Time",
        "Seconds",
        log_scale=True,
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}speed.png")

    # --- Scalability metrics
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(axes[0], "parameters", "Parameter Count", "Count")
    _plot_metric_group(axes[1], "infer_flops", "Inference FLOPs per Sample", "GFLOPs")
    _plot_metric_group(axes[2], "train_flops", "Training FLOPs per Sample", "GFLOPs")
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}flops.png")

    # --- Performance metrics: Row 1
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(
        axes[0],
        "MSE(y_hat_gp, y_hat)",
        "MSE Mean Prediction vs. GP Mean Prediction",
        "MSE",
        loc="upper left",
    )
    _plot_metric_group(
        axes[1],
        "ls wasserstein distance",
        "Lengthscale Wasserstein Distance",
        "Distance",
        loc="upper left",
    )
    _plot_metric_group(
        axes[2],
        "ESS ls",
        "Lengthscale ESS",
        "ESS",
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}performance_row1.png")

    # --- Performance metrics: Row 2
    _, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    _plot_metric_group(
        axes[0],
        "MSE(y, y_hat)",
        "Mean Prediction MSE (All Locations)",
        "MSE",
        loc="upper left",
    )
    _plot_metric_group(
        axes[1],
        "obs MSE(y, y_hat)",
        "Mean Prediction MSE (Observed Locations)",
        "MSE",
        loc="upper left",
    )
    _plot_metric_group(
        axes[2],
        "unobs MSE(y, y_hat)",
        "Mean Prediction MSE (Unobserved Locations)",
        "MSE",
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}performance_row2.png")


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


def aggregate_csvs(base_path: Path):
    levels = [16**2, 24**2, 32**2, 48**2, 64**2]
    df_list = []
    for L in levels:
        file_path = (base_path / f"grid_{L}") / "res.csv"
        df = pd.read_csv(file_path)
        df_list.append(df)
    aggregated_df = pd.concat(df_list, ignore_index=True)
    aggregated_df.drop(
        columns=[k for k in aggregated_df.columns if k.startswith("Unna")], inplace=True
    )
    output_path = base_path / "aggregated_results.csv"
    aggregated_df.to_csv(output_path, index=False)
    print(f"Aggregated results saved to {output_path}")
    return aggregated_df


if __name__ == "__main__":
    main(seed=42, gt_ls=10)
    main(seed=78, gt_ls=30)
    # NOTE: I messed up the rngs in gt_ls=50, and don't want to rerun GP (71h for 4096 grid)
    # NOTE: Running gt_ls in these two intervals breaks the rngs correctly to reproduce results
    main(seed=34, gt_ls=50, grids=[24, 48, 64])
    main(seed=34, gt_ls=50, grids=[16, 1, 32])
