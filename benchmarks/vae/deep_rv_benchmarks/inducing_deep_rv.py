import sys

sys.path.append("benchmarks/vae")
import argparse
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
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from sklearn.cluster import KMeans
from sps.kernels import matern_1_2
from sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(seed=51, logged_priors=True, solve_inv: bool = False):
    wandb.init(mode="disabled")  # NOTE: downstream function assumes active wandb
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path(f"results/inducing_drv{'_log_priors' if logged_priors else ''}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 256}] * 2).reshape(-1, 2)
    models = {
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        # "Baseline_GP": None,
        "Inducing Points": None,
    }
    y_obs = gen_y_obs(rng_obs, s)
    obs_mask = generate_obs_mask(rng_idxs, y_obs)
    priors = {
        "ls": dist.Uniform(1.0, 100.0),
        "log_ls": dist.Beta(4.0, 1.0),
        "beta": dist.Normal(),
    }

    y_hats, all_samples, result = [], [], []
    L_train, L = int(s.shape[0] ** 1.75), s.shape[0]
    for model_name, nn_model in models.items():
        infer_model, cond_names, s_train = inference_model_inducing_points(
            s, priors, obs_mask, logged_priors, L_train, solve_inv
        )
        loader = gen_train_dataloader(
            s_train, priors, logged_priors, solve_inv, batch_size=32
        )
        train_time, eval_mse, surrogate_decoder, ess = None, None, None, {}
        if nn_model is not None:
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train, rng_test, loader, nn_model
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
                "infer_time": infer_time,
                "inferred lengthscale mean": samples["ls"].mean(axis=0),
                "inferred fixed effects": samples["beta"].mean(axis=0),
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "ESS spatial effects": ess["mu"].mean().item() if ess else None,
                "ESS lengthscale": ess["ls"].item() if ess else None,
                "ESS fixed effects": ess["beta"].item() if ess else None,
            }
        )
    model_names = list(models.keys())
    plot_posterior_predictive_comparisons(
        all_samples,
        {},
        priors,
        model_names,
        cond_names,
        save_dir / "comp",
        "Inducing Points",
    )
    plot_models_predictive_means(
        y_obs, y_hats, obs_mask, model_names, save_dir / "obs_means.png"
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


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    train_num_steps: int = 200_000,
    valid_interval: int = 50_000,
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
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(
    s_train: Array, priors: dict, logged_priors: bool, solve_inv: bool, batch_size: int
):
    jitter = 5e-4 * jnp.eye(s_train.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_1_2(s, s, var, ls) + jitter)
    f_jit = jit(lambda K, z: jnp.einsum("ij,bj->bi", jnp.linalg.cholesky(K), z))

    def dataloader(rng_data):
        while True:
            rng_data, rng_ls, rng_z = random.split(rng_data, 3)
            var = 1.0
            if logged_priors:
                ls = jnp.exp(priors["log_ls"].sample(rng_ls) * jnp.log(100))
            else:
                ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = kernel_jit(s_train, var, ls)
            f = f_jit(K, z)
            if solve_inv:
                f = jnp.linalg.solve(K, f)
            yield {"s": s_train, "f": f, "z": z, "conditionals": jnp.array([ls])}

    return dataloader


def inference_model_inducing_points(
    s: Array,
    priors: dict,
    obs_mask: Array,
    logged_priors: bool,
    num_points: int,
    solve_inv: bool,
):
    """Builds a poisson likelihood inference model for inducing points"""
    kmeans = KMeans(n_clusters=num_points, random_state=0)
    u = kmeans.fit(s[obs_mask]).cluster_centers_  # shape (num_points, s.shape[1])
    surrogate_kwargs = {"s": u}  # we train deepRV with u

    def poisson_inducing(surrogate_decoder=None, obs_mask=True, y=None):
        var = 1.0
        if logged_priors:
            log_ls = numpyro.sample("log_ls", priors["log_ls"], sample_shape=())
            ls = numpyro.deterministic("ls", jnp.exp(log_ls * jnp.log(100)))
        else:
            ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        K_uu = matern_1_2(u, u, var, ls) + 5e-4 * jnp.eye(u.shape[0])
        K_su = matern_1_2(s, u, var, ls)
        if surrogate_decoder is None:
            f_u = numpyro.sample("mu", dist.MultivariateNormal(0.0, K_uu))
            s_K_uu = jnp.linalg.solve(K_uu, f_u)
        else:
            z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
            s_K_uu = surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze()
            if not solve_inv:
                s_K_uu = jnp.linalg.solve(K_uu, s_K_uu)
        f = numpyro.deterministic("f", K_su @ s_K_uu)
        # NOTE: uncomment to perform FITC correction for marginal variances
        # K_uu_inv_K_us = jnp.linalg.solve(K_uu, K_su.T)
        # Q_ss_diag = jnp.sum(K_su * K_uu_inv_K_us.T, axis=1)
        # delta = jnp.clip(var - Q_ss_diag, 1.0e-6, jnp.inf)  # K_ss_diag = var
        # f_mu = numpyro.sample("f_mu", dist.Normal(f_mu, jnp.sqrt(delta)).to_event(1))
        lambda_ = jnp.exp(f + beta)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(lambda_), obs=y)

    return poisson_inducing, priors.keys(), u


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


def generate_obs_mask(rng: Array, y_obs: Array, obs_ratio: float = 0.2):
    """Creates a mask which indicates to the inference model which locations to
    observe. Randomly chooses a subset of location to be observed."""
    L = y_obs.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.isin(jnp.arange(L), obs_idxs)


def plot_models_predictive_means(
    s,
    s_train,
    f_obs,
    f_hats,
    obs_mask,
    model_names,
    save_path: Path,
    log=True,
):
    f_hat_means = [f_mean.mean(axis=0).reshape(64, 64) for f_mean in f_hats]
    f_obs = f_obs.reshape(64, 64)
    if log:
        f_hat_means = [jnp.log(f + 1) for f in f_hat_means]
        f_obs = jnp.log(f_obs + 1)
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    cols = 5
    rows = int(jnp.ceil((len(f_hat_means) + 3) / cols))
    fig, ax = plt.subplots(
        rows, cols, figsize=(6 * cols, 7 * rows), constrained_layout=True
    )
    ax = ax.flatten()
    distances_sq = jnp.sum((s[:, None, :] - s_train[None, :, :]) ** 2, axis=-1)
    closest_s_indices = jnp.argmin(distances_sq, axis=0)
    closest_mask = jnp.zeros(s.shape[0], dtype=bool)
    closest_mask = closest_mask.at[closest_s_indices].set(True)
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(64, 64), f_obs)
    f_train = np.ma.masked_where(~closest_mask.reshape(64, 64), f_obs)
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    ax[0].set_title("y observed")
    ax[1].imshow(f_train, origin="lower", cmap=cmap)
    ax[1].set_title("y train")
    ax[2].imshow(f_obs, vmin=vmin, vmax=vmax, origin="lower")
    ax[2].set_title("y")
    for i, f_mean in enumerate(f_hat_means, start=3):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--solve_inv", action="store_true")
    args = parser.parse_args()
    main(solve_inv=args.solve_inv)
