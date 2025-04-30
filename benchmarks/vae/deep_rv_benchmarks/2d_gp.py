import sys

sys.path.append("benchmarks/vae")
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
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.optim import Adam
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from sklearn.cluster import KMeans
from sps.kernels import matern_5_2
from sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import PriorCVAE, TransformerDeepRV, gMLPDeepRV
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    generate_surrogate_decoder,
    prior_cvae_train_step,
)


def main(seed=42, logged_priors=True):
    wandb.init(mode="disabled")  # NOTE: downstream function assume active wandb
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path(f"results/2d_gp{'_log_priors' if logged_priors else ''}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 32}] * 2).reshape(-1, 2)
    L = s.shape[0]
    models = {
        "Baseline_GP": None,
        "PriorCVAE": PriorCVAE(MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L),
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "DeepRV + Transfomer": TransformerDeepRV(num_blks=2, dim=64),
        "ADVI": None,
        "Inducing Points": None,
    }
    y_obs = gen_y_obs(rng_obs, s)
    obs_mask = generate_obs_mask(rng_idxs, y_obs)
    # NOTE: LogUniform is used in the approximate inference case - ADVI, inducing points
    priors = {
        "ls": dist.Uniform(1.0, 100.0),
        "log_ls": dist.Beta(4.0, 1.0),
        "beta": dist.Normal(),
    }
    poisson_llk, cond_names = inference_model(s, priors, logged_priors)
    poisson_inducing_llk = inference_model_inducing_points(
        s, priors, obs_mask, logged_priors, 64
    )
    loader = gen_train_dataloader(s, priors, logged_priors)
    y_hats, all_samples, result = [], [], []
    for model_name, nn_model in models.items():
        infer_model = (
            poisson_inducing_llk if model_name == "Inducing Points" else poisson_llk
        )
        train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
        if nn_model is not None:
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train, rng_test, loader, model_name, nn_model
            )
        if model_name != "ADVI":
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
        else:
            samples, post, infer_time = advi(rng_infer, infer_model, y_obs)
        y_hats.append(post["obs"])
        all_samples.append(samples)
        result.append(
            {
                "model_name": model_name,
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "infer_time": infer_time,
                "inferred lengthscale mean": samples["ls"].mean(axis=0),
                "inferred fixed effects": samples["beta"].mean(axis=0),
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "ESS spatial effects": ess["mu"].mean().item() if ess else None,
                "ESS lengthscale": ess["ls"].item() if ess else None,
                "ESS fixed effects": ess["beta"].item() if ess else None,
            }
        )
    plot_posterior_predictive_comparisons(
        all_samples, {}, priors, list(models.keys()), cond_names, save_dir / "comp"
    )
    plot_models_predictive_means(
        y_obs, y_hats, obs_mask, list(models.keys()), save_dir / "obs_means.png"
    )
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
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
    post = Predictive(model, samples)(k2)
    return samples, mcmc, post, infer_time


def advi(rng: Array, model: Callable, y_obs: Array, num_steps=50_000):
    rng_advi, rng_pp, rng_post = random.split(rng, 3)
    guide = AutoMultivariateNormal(model)
    optimizer = Adam(step_size=0.0001)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    start = datetime.now()
    svi_result = svi.run(rng_advi, num_steps, y=y_obs, stable_update=True)
    infer_time = (datetime.now() - start).total_seconds()
    params = svi_result.params
    samples = guide.sample_posterior(rng_pp, params, sample_shape=(40_000,))
    post = Predictive(model, samples)(rng_post)
    return samples, post, infer_time


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
    lr_schedule = cosine_annealing_lr(train_num_steps, 1.0e-3)
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
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(s: Array, priors: dict, logged_priors: bool, batch_size=32):
    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            if logged_priors:
                ls = jnp.exp(priors["log_ls"].sample(rng_ls) * jnp.log(100))
            else:
                ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            chol = jnp.linalg.cholesky(K)
            f = jnp.einsum("ij,bj->bi", chol, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


def inference_model(s: Array, priors: dict, logged_priors: bool):
    """
    Builds a poisson likelihood inference model for GP and surrogate models
    """
    surrogate_kwargs = {"s": s}

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        if logged_priors:
            log_ls = numpyro.sample("log_ls", priors["log_ls"], sample_shape=())
            ls = numpyro.deterministic("ls", jnp.exp(log_ls * jnp.log(100)))
        else:
            ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        else:
            K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            L_chol = jnp.linalg.cholesky(K)
            mu = numpyro.deterministic("mu", jnp.matmul(L_chol, z[0]))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return poisson, ["ls", "beta"]


def inference_model_inducing_points(
    s: Array, priors: dict, obs_mask: Array, logged_priors: bool, num_points: int
):
    """Builds a poisson likelihood inference model for inducing points"""
    kmeans = KMeans(n_clusters=num_points, random_state=0)
    u = kmeans.fit(s[obs_mask]).cluster_centers_  # shape (num_points, s.shape[1])

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        if logged_priors:
            log_ls = numpyro.sample("log_ls", priors["log_ls"], sample_shape=())
            ls = numpyro.deterministic("ls", jnp.exp(log_ls * jnp.log(100)))
        else:
            ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_uu = matern_5_2(u, u, var, ls) + 5e-4 * jnp.eye(u.shape[0])
        K_su = matern_5_2(s, u, var, ls)
        f_u = numpyro.sample("mu", dist.MultivariateNormal(0.0, K_uu))
        f = K_su @ jnp.linalg.solve(K_uu, f_u)
        # NOTE: uncomment to perform FITC correction for marginal variances
        # K_uu_inv_K_us = jnp.linalg.solve(K_uu, K_su.T)
        # Q_ss_diag = jnp.sum(K_su * K_uu_inv_K_us.T, axis=1)
        # delta = jnp.clip(var - Q_ss_diag, 1.0e-6, jnp.inf)  # K_ss_diag = var
        # f_mu = numpyro.sample("f_mu", dist.Normal(f_mu, jnp.sqrt(delta)).to_event(1))
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


def gen_y_obs(rng: Array, s: Array):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, 10.0, 1.0
    K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def generate_obs_mask(rng: Array, y_obs: Array, obs_ratio: float = 0.15):
    """Creates a mask which indicates to the inference model which locations to
    observe. Randomly chooses a subset of location to be observed."""
    L = y_obs.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.array([i in obs_idxs for i in range(L)])


def plot_models_predictive_means(
    f_obs, f_hats, obs_mask, model_names, save_path: Path, log=True
):
    f_hat_means = [f_mean.mean(axis=0).reshape(32, 32) for f_mean in f_hats]
    f_obs = f_obs.reshape(32, 32)
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
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(32, 32), f_obs)
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    ax[0].set_title("y observed")
    ax[1].imshow(f_obs, vmin=vmin, vmax=vmax, origin="lower")
    ax[1].set_title("y")
    for i, f_mean in enumerate(f_hat_means, start=2):
        model_name = model_names[i - 2]
        im = ax[i].imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower")
        ax[i].set_title("Mean " r"$\hat{y}$" f" {model_name}")
    for i in range(len(ax)):
        ax[i].set_axis_off()
        if (i + 1) % cols == 0:
            fig.colorbar(im, ax=ax[i])
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    main()
