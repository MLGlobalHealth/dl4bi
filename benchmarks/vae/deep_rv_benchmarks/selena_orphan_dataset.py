import sys

sys.path.append("benchmarks/vae")
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union

import arviz as az
import flax.linen as nn
import geopandas as gpd
import jax.numpy as jnp
import numpyro
import optax
import pandas as pd
from jax import Array, jit, random
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from reproduce_paper.deep_rv_plots import (
    plot_models_predictive_means,
    plot_posterior_predictive_comparisons,
)
from shapely.affinity import scale, translate
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, rbf
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, train
from dl4bi.vae import MLPDeepRV, TransformerDeepRV, gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(seed=42):
    wandb.init(mode="disabled")
    rng = random.key(seed)
    rng_train, rng_test, rng_infer = random.split(rng, 3)
    kernel = matern_1_2  # NOTE: choose from: matern_1_2, matern_3_2, matern_5_2, rbf
    save_dir = Path(f"results/orphan_dataset/{kernel.__name__}")
    save_dir.mkdir(parents=True, exist_ok=True)
    # NOTE: change to the correct path for you
    map_data = gpd.read_file("benchmarks/vae/maps/Selena_2k_grid")
    # NOTE: specifically for 2k grid - these location are an outlier
    map_data = map_data.drop(index=[167, 168, 169]).reset_index(drop=True)
    s = gen_spatial_structure(map_data)
    L = s.shape[0]
    models = {
        "Baseline_GP": None,  # Baseline test - can remove if takes too long
        "DeepRV + MLP": MLPDeepRV(dims=[L, L]),
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "DeepRV + Transfomer": TransformerDeepRV(num_blks=2, dim=64),
    }
    # NOTE: change to correct data columns paternal\maternal etc
    y_obs = jnp.array(map_data.paternal, dtype=jnp.float32)
    population = jnp.array(map_data.childtotal, dtype=jnp.int32)
    obs_mask = jnp.array(map_data.observed == 1, dtype=jnp.bool)
    priors = {
        "var": dist.Gamma(1.5, 1.5),
        "ls": dist.InverseGamma(3.0, 3.0),
        "beta": dist.Normal(),  # NOTE: intercept
    }
    binom_infer_model, cond_names = inference_model(s, priors, kernel, population)
    loader = gen_train_dataloader(s, priors, kernel)
    y_hats, all_samples, result = [y_obs], [], []
    for model_name, model in models.items():
        optimizer = optax.yogi(cosine_annealing_lr(100_000, 5.0e-3, lr_min=0.0))
        if model_name == "DeepRV + ScanTransfomer":
            optimizer = optax.adamw(5.0e-4)
        train_time, eval_mse, surrogate_decoder = None, None, None
        if model_name != "Baseline_GP":
            # NOTE: Train the surrogate
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train, rng_test, loader, model, optimizer
            )
        # NOTE: Run inference
        samples, mcmc, post, infer_time = hmc(
            rng_infer, binom_infer_model, y_obs, surrogate_decoder, obs_mask
        )
        y_hats.append(post["obs"])
        all_samples.append(samples)
        ess = az.ess(mcmc, method="mean")
        plot_infer_trace(
            samples, mcmc, None, cond_names, save_dir / f"{model_name}_infer_trace.png"
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
                "ESS spatial effects": ess["mu"].mean().item(),
                "ESS lengthscale": ess["ls"].item(),
                "ESS variance": ess["var"].item(),
                "ESS fixed effects": ess["beta"].item(),
            }
        )
    plot_posterior_predictive_comparisons(
        all_samples, {}, priors, list(models.keys()), cond_names, save_dir / "comp"
    )
    # NOTE: plots the observed means by the order of the models above
    plot_models_predictive_means(y_hats, map_data, save_dir / "obs_means.png")
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    surrogate_decoder: Optional[Callable] = None,
    obs_mask: Union[Array, bool] = True,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=4, num_samples=10_000, num_warmup=4_000)
    # mcmc = MCMC(nuts, num_chains=1, num_samples=10_00, num_warmup=4_00)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, y=y_obs, obs_mask=obs_mask)
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
    optimizer,
    train_num_steps: int = 100_000,
    valid_interval: int = 25_000,
    valid_steps: int = 5_000,
):
    start = datetime.now()
    state = train(
        rng_train,
        model,
        optimizer,
        partial(deep_rv_train_step, var_idx=0),
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


def gen_train_dataloader(s: Array, priors: dict, kernel: Callable, batch_size=32):
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


def inference_model(s: Array, priors: dict, kernel: Callable, population: Array):
    """
    Builds a Binomial inference model for either actual GP or a surrogate.

    Args:
        s: Locations (n, dim_s).
        population: array of population per location N

    Returns:
        A NumPyro model function, and the parameter names
    """
    surrogate_kwargs = {"s": s}

    def binomial(surrogate_decoder=None, obs_mask=True, y=None):
        var = numpyro.sample("var", priors["var"], sample_shape=())
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=())
        if surrogate_decoder:  # whether to use a replacment for the GP
            z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
            mu = numpyro.deterministic(
                "mu",
                surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        else:
            K = kernel(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.sample("mu", dist.MultivariateNormal(0.0, K))
        eta = jnp.sqrt(var) * mu + beta
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Binomial(population, logits=eta), obs=y)

    return binomial, ["var", "ls", "beta"]


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


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


if __name__ == "__main__":
    main()
