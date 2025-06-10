import sys

sys.path.append("benchmarks/vae")
import pickle
from datetime import datetime
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
from omegaconf import DictConfig
from reproduce_paper.deep_rv_plots import (
    plot_models_predictive_means,
    plot_posterior_predictive_comparisons,
)
from shapely.affinity import scale, translate
from sps.kernels import matern_3_2
from utils.plot_utils import plot_infer_trace

import wandb
from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae import gMLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step, generate_surrogate_decoder


def main(seed=63):
    wandb.init(mode="disabled")
    rng = random.key(seed)
    rng_train, rng_test, rng_infer, rng_idxs = random.split(rng, 4)
    save_dir = Path("results/pm10_msoa/")
    save_dir.mkdir(parents=True, exist_ok=True)
    map_data = gpd.read_file("benchmarks/vae/maps/msoa_with_pm10_aggregated")
    map_data = map_data[~map_data.pm102023g_.isna()]
    s, norm_areas = gen_spatial_structure(map_data, s_max=100)
    models = {
        "DeepRV + gMLP": gMLPDeepRV(num_blks=2),
        "Baseline_GP": None,
    }
    y_obs = jnp.array(map_data.pm102023g_, dtype=jnp.float32)
    obs_mask = generate_obs_mask(rng_idxs, y_obs)
    priors = {
        "var": dist.Gamma(1.5, 1.5),
        "ls": dist.Uniform(1.0, 50.0),  # s is scaled 0-100
        "beta_1": dist.Normal(),
        "beta_0": dist.Normal(),
        "obs_noise": dist.HalfNormal(),
    }
    infer_model, cond_names = inference_model(s, priors, jnp.array(norm_areas))
    loader = gen_train_dataloader(s, priors)
    y_hats, all_samples, result = [y_obs], [], []
    for model_name, model in models.items():
        (save_dir / f"{model_name}").mkdir(parents=True, exist_ok=True)
        train_time, eval_mse, surrogate_decoder = None, None, None
        if model_name != "Baseline_GP":
            train_time, eval_mse, surrogate_decoder = surrogate_model_train(
                rng_train, rng_test, loader, model, save_dir / f"{model_name}"
            )
        samples, mcmc, post, infer_time = hmc(
            rng_infer,
            infer_model,
            y_obs,
            obs_mask,
            save_dir / f"{model_name}",
            surrogate_decoder,
        )
        y_hats.append(post["obs"])
        all_samples.append(samples)
        ess = az.ess(mcmc, method="mean")
        plot_infer_trace(
            samples, mcmc, None, cond_names, save_dir / f"{model_name}_infer_trace.png"
        )
        res = {
            "model_name": model_name,
            "train_time": train_time,
            "Test Norm MSE": eval_mse,
            "infer_time": infer_time,
            "MSE(y, y_hat)": ((y_obs - post["obs"].mean(axis=0)) ** 2).mean(),
        }
        res.update({f"inferred {c} mean": samples[c].mean(axis=0) for c in cond_names})
        res.update(
            {f"ESS {c}": ess[c].mean().item() if ess else None for c in cond_names}
        )
        result.append(res)
    plot_posterior_predictive_comparisons(
        all_samples, {}, priors, list(models.keys()), cond_names, save_dir / "comp"
    )
    plot_models_predictive_means(
        y_hats, map_data, save_dir / "obs_means.png", obs_mask, log=False
    )
    pd.DataFrame(result).to_csv(save_dir / "res.csv")


def hmc(
    rng: Array,
    model: Callable,
    y_obs: Array,
    obs_mask: Union[Array, bool],
    results_dir: Path,
    surrogate_decoder: Optional[Callable] = None,
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = random.split(rng)
    mcmc = MCMC(nuts, num_chains=1, num_samples=1000, num_warmup=1000)
    start = datetime.now()
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, y=y_obs, obs_mask=obs_mask)
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


def surrogate_model_train(
    rng_train: Array,
    rng_test: Array,
    loader: Callable,
    model: nn.Module,
    results_dir: Path,
    train_num_steps: int = 300_000,
    valid_interval: int = 50_000,
    valid_steps: int = 5_000,
):
    train_step = deep_rv_train_step
    optimizer = optax.chain(
        optax.clip_by_global_norm(3.0),
        optax.yogi(cosine_annealing_lr(train_num_steps, 1e-2)),
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
        pickle.dump([train_time], out_file)
    surrogate_decoder = generate_surrogate_decoder(state, model)
    return train_time, eval_mse, surrogate_decoder


def gen_train_dataloader(s: Array, priors: dict[str, dist.Distribution], batch_size=16):
    jitter = 5e-4 * jnp.eye(s.shape[0])
    kernel_jit = jit(lambda s, var, ls: matern_3_2(s, s, var, ls) + jitter)
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


def inference_model(s: Array, priors: dict, norm_areas: Array):
    """
    Builds a Binomial inference model for either actual GP or a surrogate.

    Args:
        s: Locations (n, dim_s).
        norm_areas: array of normalized areas per location

    Returns:
        A NumPyro model function, and the parameter names
    """
    surrogate_kwargs = {"s": s}

    def log_normal(surrogate_decoder=None, obs_mask=True, y=None):
        var = numpyro.sample("var", priors["var"], sample_shape=())
        ls = numpyro.sample("ls", priors["ls"], sample_shape=())
        beta_0 = numpyro.sample("beta_0", priors["beta_0"], sample_shape=())
        beta_1 = numpyro.sample("beta_1", priors["beta_1"], sample_shape=())
        obs_noise = numpyro.sample("obs_noise", priors["obs_noise"], sample_shape=())
        if surrogate_decoder:  # whether to use a replacment for the GP
            z = numpyro.sample("z", dist.Normal(), sample_shape=(1, s.shape[0]))
            mu = numpyro.deterministic(
                "mu",
                jnp.sqrt(var)
                * surrogate_decoder(z, jnp.array([ls]), **surrogate_kwargs).squeeze(),
            )
        else:
            K = matern_3_2(s, s, var, ls) + 5e-4 * jnp.eye(s.shape[0])
            mu = numpyro.sample("mu", dist.MultivariateNormal(0.0, K))
        eta = mu + beta_0 + beta_1 * norm_areas
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.LogNormal(eta, obs_noise), obs=y)

    return log_normal, list(priors.keys())


@jit
def valid_step(rng, state, batch):
    output: VAEOutput = state.apply_fn(
        {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
    )
    metrics = output.metrics(batch["f"], 1.0)
    return {"norm MSE": metrics["MSE"]}


def gen_spatial_structure(map_data: gpd.GeoDataFrame, s_max: int):
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
    norm_areas = map_data.geometry.area
    centroids = norm_map.geometry.centroid
    return jnp.stack([centroids.x.values, centroids.y.values], axis=-1), norm_areas


def generate_obs_mask(rng: Array, y_obs: Array, obs_ratio: float = 0.2):
    """Generate a boolean mask selecting a subset of locations to observe."""
    L = y_obs.shape[0]
    num_obs_locations = int(obs_ratio * L)
    obs_idxs = random.choice(rng, L, shape=(num_obs_locations,), replace=False)
    return jnp.isin(jnp.arange(L), obs_idxs)


if __name__ == "__main__":
    main()
