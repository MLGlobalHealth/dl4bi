import sys
from functools import partial

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
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from reproduce_paper.deep_rv_plots import plot_posterior_predictive_comparisons
from sps.kernels import matern_5_2
from sps.utils import build_grid
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.mlp import MLP, gMLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import DeepRV, PriorCVAE
from dl4bi.vae.train_utils import (
    cond_as_feats,
    cond_as_locs,
    deep_rv_train_step,
    generate_surrogate_decoder,
    prior_cvae_train_step,
)


def main(seed=42):
    wandb.init(mode="disabled")
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs, rng_obs = random.split(rng, 5)
    save_dir = Path("results/2d_gp/")
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 32}] * 2).reshape(-1, 2)
    L = s.shape[0]
    models = {
        "Baseline_GP": None,
        "PriorCVAE": PriorCVAE(MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L),
        "DeepRV": DeepRV(gMLP(num_blks=2), cond_as_feats),
    }
    y_obs = gen_y_obs(rng_obs, s)
    priors = {
        "var": dist.Gamma(1.5, 1.5),
        "ls": dist.Uniform(0.0, 50.0),
        "beta": dist.Normal(),
    }
    poisson_infer_model, cond_names = inferece_model(s, priors)
    loader = gen_train_dataloader(s, priors)
    obs_mask = generate_obs_mask(rng_idxs, y_obs)
    y_hats, all_samples, result = [y_obs], [], []
    for model_name, model in models.items():
        train_time, eval_mse, surrogate_decoder = None, None, None
        if model_name != "Baseline_GP":
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train,
                rng_test,
                loader,
                model_name,
                model,
            )
        samples, mcmc, post, infer_time = _hmc(
            rng_infer,
            poisson_infer_model,
            y_obs,
            obs_mask,
            surrogate_decoder,
        )
        y_hats.append(post["obs"])
        all_samples.append(samples)
        ess = az.ess(mcmc, method="mean")
        plot_infer_trace(
            samples,
            mcmc,
            conditionals=None,
            var_names=cond_names,
            save_path=save_dir / f"{model_name}_infer_trace.png",
        )
        result.append(
            {
                "model_name": model_name,
                "train_time": train_time,
                "Test Norm MSE": eval_mse,
                "infer_time": infer_time,
                "inferred lengthscale mean": samples["ls"].mean(axis=0),
                "inferred fixed effects": samples["beta"].mean(axis=0),
                "inferred variance": samples["var"].mean(axis=0),
                "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
                "ESS spatial effects": ess["mu"],
                "ESS lengthscale": ess["ls"],
                "ESS variance": ess["mu"],
                "ESS fixed effects": ess["beta"],
            }
        )
    plot_posterior_predictive_comparisons(
        all_samples,
        {n: None for n in cond_names},
        priors,
        list(models.keys()),
        cond_names,
        save_prefix=save_dir / "comp",
    )
    plot_models_predictive_means(y_hats, obs_mask, save_dir / "obs_means.png")
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def _hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[bool, Array],
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
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
    model_name: str,
    model: nn.Module,
    train_num_steps: int = 100_000,
):
    train_step = prior_cvae_train_step
    lr_schedule = cosine_annealing_lr(train_num_steps, 1.0e-3)
    if model_name != "PriorCVAE":
        train_step = partial(deep_rv_train_step, var_idx=0)
        lr_schedule = cosine_annealing_lr(train_num_steps, 1.0e-2)
    optimizer = optax.chain(optax.clip_by_global_norm(3.0), optax.yogi(lr_schedule))
    start = datetime.now()
    state = train(rng_train, model, optimizer, train_step, train_num_steps, loader)
    train_time = (datetime.now() - start).total_seconds()
    eval_mse = evaluate(rng_test, state, valid_step, loader, 5_000)["norm MSE"]
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(s: Array, priors: dict, batch_size=32):
    def dataloader(rng_data):
        while True:
            rng_data, rng_var, rng_ls, rng_z = random.split(rng_data, 4)
            var = priors["var"].sample(rng_var)
            ls = priors["ls"].sample(rng_ls)
            z = dist.Normal().sample(rng_z, sample_shape=(batch_size, s.shape[0]))
            K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            L = jnp.linalg.cholesky(K)
            f = jnp.einsum("ij,bj->bi", L, z)
            yield {"s": s, "f": f, "z": z, "conditionals": jnp.array([var, ls])}

    return dataloader


def inferece_model(s: Array, priors: dict):
    """
    Builds an inference model for both GP baseline and surrogate inference.

    Args:
        s: Locations (n, dim_s).
        priors: Dictionary of prior distributions.

    Returns:
        A NumPyro model function.
    """

    def poisson(surrogate_decoder=None, obs_mask=True, y=None):
        var = numpyro.sample("var", priors["var"], sample_shape=())
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        if surrogate_decoder:  # NOTE: whether to use a replacment for the GP
            z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
            mu = numpyro.deterministic(
                "mu", surrogate_decoder(z, jnp.array([var, ls])).squeeze()
            )
        else:
            K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.sample("mu", dist.MultivariateNormal(0.0, K))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            f = numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)
        return f, mu, jnp.array([var, ls, beta])

    return poisson, ["var", "ls", "beta"]


@jit
def valid_step(rng, state, batch):
    f, conditionals = batch["f"], batch["conditionals"]
    var = conditionals[0].squeeze()
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(f, var)
    return {"norm MSE": (1 / var) * metrics["MSE"]}


def gen_y_obs(rng: Array, s: Array):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, 10.0, 1.0
    K = matern_5_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def generate_obs_mask(rng: Array, y_obs: Array, obs_ratio: float = 0.2):
    """Creates a mask which indicates to the inference model which locations to
    observe. Randomly chooses a subset of all the valid locations
    are randmoly chosen to be observed.

    Args:
        rng (Array)
        f_obs (Array): The sample to infer from
        obs_ratio (float): ratio of location to take

    Returns:
        (Union[Array, bool], Array): boolean observe mask and
            indices of the observed locations
    """
    L = y_obs.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.array([i in obs_idxs for i in range(L)])


def plot_models_predictive_means(f_hats, obs_mask, save_path: Path):
    f_hat_means = [f_hats[0]] + [f_mean.mean(axis=0) for f_mean in f_hats[1:]]
    f_hat_means = [f_hat.reshape(32, 32) for f_hat in f_hat_means]
    vmin = jnp.min(jnp.array([f_mean.min() for f_mean in f_hat_means])).item()
    vmax = jnp.max(jnp.array([f_mean.max() for f_mean in f_hat_means])).item()
    cols = 3
    rows = int(jnp.ceil((len(f_hat_means) + 1) / 3))
    fig, ax = plt.subplots(
        rows, cols, figsize=(6 * cols, 7 * rows), constrained_layout=True
    )
    ax = ax.flatten()
    masked_f_obs = np.ma.masked_where(~obs_mask.reshape(32, 32), f_hat_means[0])
    cmap = plt.cm.viridis
    cmap.set_bad(color="black")
    ax[0].imshow(masked_f_obs, origin="lower", cmap=cmap)
    for i, f_mean in enumerate(f_hat_means):
        im = ax[i + 1].imshow(f_mean, vmin=vmin, vmax=vmax, origin="lower")
        if i == len(f_hat_means) - 1:
            fig.colorbar(im, ax=ax[i + 1])
    for i in range(len(ax)):
        ax[i].set_axis_off()
    fig.savefig(save_path, dpi=200)
    plt.clf()
    plt.close(fig)


if __name__ == "__main__":
    main()
