import sys

sys.path.append("benchmarks/vae")
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import geopandas as gpd
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from scipy.stats import wasserstein_distance
from dl4bi_sps.kernels import matern_1_2, matern_3_2, matern_5_2, rbf
from dl4bi_sps.utils import build_grid
from utils.plot_utils import conds_to_title, plot_on_map

import wandb
from dl4bi.core.mlp import MLP
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import (
    FixedKernelAttention,
    KernelBiasTransformerDeepRV,
    MLPDeepRV,
    PriorCVAE,
    gMLPDeepRV,
)
from dl4bi.vae.train_utils import (
    cond_as_locs,
    deep_rv_train_step,
    generate_surrogate_decoder,
    prior_cvae_train_step,
)


def main(init_seed=42, num_seeds=3):
    wandb.init(mode="disabled")  # NOTE: downstream function assume active wandb
    save_dir = Path("results/ablation_test/")
    save_dir.mkdir(parents=True, exist_ok=True)
    s = build_grid([{"start": 0.0, "stop": 100.0, "num": 24}] * 2).reshape(-1, 2)
    L = s.shape[0]
    models = {
        "GP": None,
        "PriorCVAE": PriorCVAE(MLP(dims=[L, L]), MLP(dims=[L, L]), cond_as_locs, L),
        "DeepRV + MLP": MLPDeepRV(dims=[L, L]),
        "DeepRV + gMLP adamw": gMLPDeepRV(num_blks=2),
        "DeepRV + gMLP kAttn": gMLPDeepRV(num_blks=2, attn=FixedKernelAttention()),
        "DeepRV + trans kAttn": KernelBiasTransformerDeepRV(max_locations=L),
    }
    kernels = [matern_1_2, matern_5_2, matern_3_2, rbf]
    priors = {"ls": dist.Uniform(1.0, 100.0), "beta": dist.Normal()}
    # NOTE: > 5 so PriorCVAE won't break, < 50 so it would have some variability
    gen_data_priors = {
        "ls": dist.Uniform(5.0, 50.0),
        "beta": dist.Uniform(low=0.6, high=2.0),
    }
    result = []
    seeds = random.choice(
        random.key(init_seed), jnp.arange(100), shape=(num_seeds,), replace=False
    )
    for seed in seeds:
        rng = random.key(seed)
        rng_obs, rng_idx, rng_infer, rng_train, rng_test, rng_beta, rng_ls = (
            random.split(rng, 7)
        )
        for kernel in kernels:
            kernel_n = kernel.__name__
            states = []
            gt_ls = float(gen_data_priors["ls"].sample(rng_ls))
            gt_beta = float(gen_data_priors["beta"].sample(rng_beta))
            obs_mask = gen_random_obs_mask(rng_idx, s, 0.4)
            y_obs = gen_y_obs(rng_obs, s, kernel, gt_ls, gt_beta)
            for model_name, nn_model in models.items():
                infer_model = inference_model(s, kernel, priors, "kAttn" in model_name)
                train_time, eval_mse, surrogate_decoder = None, None, None
                if nn_model is not None:
                    loader = gen_gp_dataloader(s, priors, kernel, "kAttn" in model_name)
                    optimizer, train_num_steps, train_step = gen_train_params(
                        model_name
                    )
                    start = datetime.now()
                    state = train(
                        rng_train,
                        nn_model,
                        optimizer,
                        train_step,
                        train_num_steps,
                        loader,
                        valid_step,
                        50_000,
                        5_000,
                        loader,
                        return_state="best",
                        valid_monitor_metric="norm MSE",
                    )
                    train_time = (datetime.now() - start).total_seconds()
                    eval_mse = evaluate(rng_test, state, valid_step, loader, 5_000)[
                        "norm MSE"
                    ]
                    surrogate_decoder = generate_surrogate_decoder(state, nn_model)
                    states.append(state)

                samples, mcmc, post, infer_time = hmc(
                    rng_infer, infer_model, y_obs, obs_mask, surrogate_decoder
                )
                n = len(samples["ls"]) // 2
                if nn_model is None:
                    gp_mean_obs = post["obs"].mean(axis=0)
                    gp_c1_mean_obs = post["obs"][:n].mean(axis=0)
                    gp_c2_mean_obs = post["obs"][n:].mean(axis=0)
                    beta_gp, ls_gp = samples["beta"], samples["ls"]
                    ls_gp_c1, ls_gp_c2 = ls_gp[:n], ls_gp[n:]
                    beta_gp_c1, beta_gp_c2 = beta_gp[:n], beta_gp[n:]
                ess = az.ess(mcmc, method="mean")
                mean_obs = post["obs"].mean(axis=0)
                model_c1_mean_obs = post["obs"][:n].mean(axis=0)
                model_c2_mean_obs = post["obs"][n:].mean(axis=0)
                ls_model_c1, ls_model_c2 = samples["ls"][:n], samples["ls"][n:]
                beta_model_c1, beta_model_c2 = samples["beta"][:n], samples["beta"][n:]
                sq_res = (y_obs - mean_obs) ** 2
                mse_cross = 0.5 * (
                    ((gp_c1_mean_obs - model_c2_mean_obs) ** 2).mean()
                    + ((gp_c2_mean_obs - model_c1_mean_obs) ** 2).mean()
                )
                ls_wass_cross = 0.5 * (
                    wasserstein_distance(ls_gp_c1, ls_model_c2)
                    + wasserstein_distance(ls_gp_c2, ls_model_c1)
                )
                beta_wass_cross = 0.5 * (
                    wasserstein_distance(beta_gp_c1, beta_model_c2)
                    + wasserstein_distance(beta_gp_c2, beta_model_c1)
                )
                result.append(
                    {
                        "model_name": model_name,
                        "train_time": train_time,
                        "Test Norm MSE": eval_mse,
                        "kernel": kernel_n,
                        "seed": seed,
                        "gt_ls": gt_ls,
                        "infer_time": infer_time,
                        "MSE(y, y_hat)": (sq_res).mean(),
                        "obs MSE(y, y_hat)": (sq_res[obs_mask]).mean(),
                        "unobs MSE(y, y_hat)": (
                            sq_res[jnp.logical_not(obs_mask)]
                        ).mean(),
                        "num_chains": 2,
                        "MSE(y_hat_gp, y_hat)": ((gp_mean_obs - mean_obs) ** 2).mean(),
                        "MSE(y_hat_gp, y_hat) cross chain": mse_cross,
                        "ls wasserstein distance": wasserstein_distance(
                            ls_gp, samples["ls"]
                        ),
                        "ls wasserstein distance cross chain": ls_wass_cross,
                        "beta wasserstein distance": wasserstein_distance(
                            beta_gp, samples["beta"]
                        ),
                        "beta wasserstein distance cross chain": beta_wass_cross,
                        "ESS ls": ess["ls"].mean().item(),
                        "ESS beta": ess["beta"].mean().item(),
                    }
                )
    result = pd.DataFrame(result)
    result.to_csv(save_dir / "res.csv")


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
    num_chains = 2
    mcmc = MCMC(nuts, num_chains=num_chains, num_samples=3_000, num_warmup=2_000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=y_obs)
    infer_time = (datetime.now() - start).total_seconds()
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2, surrogate_decoder=surrogate_decoder)
    return samples, mcmc, post, infer_time


def inference_model(s: Array, kernel: Callable, priors: dict, kernel_attn: bool):
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
            K = kernel(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
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


def gen_gp_dataloader(
    s: Array, priors: dict, kernel: Callable, kAttn: bool, batch_size=32
):
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
            batch = {"s": s, "f": f, "z": z, "conditionals": jnp.array([ls])}
            if kAttn:
                batch["K"] = K
            yield batch

    return dataloader


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_train_params(model_name):
    train_num_steps = 200_000
    max_lr = {
        "DeepRV + gMLP adamw": 1e-3,
        "DeepRV + trans": 1e-4,
        "DeepRV + trans kAttn": 1e-4,
        "DeepRV + trans kAttn no ID": 1e-4,
    }.get(model_name, 1e-3)
    train_step = {
        "PriorCVAE": prior_cvae_train_step,
    }.get(model_name, deep_rv_train_step)
    lr_schedule = cosine_annealing_lr(train_num_steps, max_lr)
    optimizer, clip = optax.adamw(lr_schedule, weight_decay=1e-2), 3.0
    if model_name in ["PriorCVAE", "DeepRV + MLP"]:
        optimizer, clip = optax.yogi(lr_schedule), 3.0
    optimizer = optax.chain(optax.clip_by_global_norm(clip), optimizer)
    return optimizer, train_num_steps, train_step


def gen_y_obs(rng: Array, s: Array, kernel: Callable, gt_ls: float, gt_beta: float):
    """generates a poisson observed data sample for inference"""
    rng_mu, rng_poiss = random.split(rng)
    var, ls, beta = 1.0, gt_ls, gt_beta
    K = kernel(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
    mu = dist.MultivariateNormal(0.0, K).sample(rng_mu)
    lambda_ = jnp.exp(beta + mu)
    return dist.Poisson(rate=lambda_).sample(rng_poiss)


def gen_random_obs_mask(rng: Array, s: Array, obs_ratio: float):
    L = s.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, jnp.arange(L), (num_obs_locations,), replace=False)
    return jnp.isin(jnp.arange(L), obs_idxs)


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


if __name__ == "__main__":
    main()
