import pickle
import sys
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import arviz as az
import geopandas as gpd
import hydra
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import optax
from jax import random
from numpyro.handlers import seed
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from utils.map_utils import gen_locations
from utils.obj_utils import generate_model_name, instantiate
from utils.plot_utils import plot_inference_run

import wandb
from dl4bi.meta_learning.train_utils import cosine_annealing_lr
from dl4bi.vae.train_utils import TrainState, generate_surrogate_decoder


@hydra.main("configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # NOTE: the model name has to match the model name used in inference
    model_name = generate_model_name(cfg)
    spatial_prior = cfg.inference_model.spatial_prior
    if not cfg.inference_model.surrogate_model:
        model_name = f"Baseline_GP_{spatial_prior.func}"
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=f"Infer_{cfg.exp_name}_{model_name}_{cfg.inference_model.model.func}",
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    rng, rng_plot, rng_idxs, rng_hmc = random.split(random.key(cfg.seed), 4)
    map_data, s = gen_locations(cfg.data)
    model_dir = Path(f"results/{cfg.exp_name}/{spatial_prior.func}/{cfg.seed}")
    priors, simulation_priors = init_priors(cfg)
    surrogate_kwargs = {}
    if "FixedLocationTransfomer" in model_name:
        surrogate_kwargs = {"s": s}
    # NOTE: the inference model expects to have this exact API
    inference_model, cond_names = gen_inference_model(
        cfg, s, map_data, priors, surrogate_kwargs
    )
    # NOTE: Generates a model for creating the simulated data
    sim_model, _ = gen_inference_model(cfg, s, map_data, simulation_priors)
    f_obs, spatial_eff, conditionals = gen_obs_data(
        rng, sim_model, map_data, cond_names
    )
    obs_mask, obs_idxs = generate_obs_mask(rng_idxs, f_obs, cfg.inference_model)
    surrogate_decoder = None
    if cfg.inference_model.surrogate_model:
        state, model = load_ckpt((model_dir / model_name).with_suffix(".ckpt"))
        surrogate_decoder = generate_surrogate_decoder(state, model)
    samples, mcmc, post = _hmc(
        rng_hmc, cfg, inference_model, f_obs, obs_mask, surrogate_decoder
    )
    post.update(
        {
            "s": s,
            "f": f_obs,
            "obs_idxs": obs_idxs,
            "spatial_eff": spatial_eff,
            **conditionals,
        }
    )
    results_dir = get_results_dir(cfg, model_dir, obs_idxs, s)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_inference_run((samples, mcmc, post), conditionals, results_dir)
    plot_inference_run(
        rng_plot,
        cfg.inference_model,
        model_name,
        (samples, mcmc, post),
        f_obs,
        conditionals,
        priors,
        map_data,
        cfg.log_scale_plots,
    )


def _hmc(
    rng: jax.Array,
    cfg: DictConfig,
    model,
    f_obs: jax.Array,
    obs_mask: Union[bool, jax.Array],
    surrogate_decoder: Optional[Callable],
):
    """runs HMC on given inference model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = jax.random.split(rng)
    mcmc = MCMC(nuts, **cfg.mcmc)
    mcmc.run(k1, surrogate_decoder=surrogate_decoder, obs_mask=obs_mask, y=f_obs)
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2)
    return samples, mcmc, post


def gen_obs_data(
    rng: jax.Array,
    sim_model: Callable,
    map_data: Optional[gpd.GeoDataFrame],
    cond_names: list[str],
):
    """Generates the observed data for inference.
    If the given dataframe has a 'data' column, the function returns it.
    Otherwise wraps GP or graph based models with an simulation model.

    Args:
        rng (jax.Array)
        sim_model (Callable): simulation model to generate data
        map_data (gpd.GeoDataFrame): original geopandas
        cond_names (list[str]): names (and order) of conditionals

    Returns:
        the inference dataloader
    """
    spatial_eff, conditionals = None, [None] * len(cond_names)
    if map_data is not None and "data" in map_data.columns:
        f_obs = jnp.array(map_data["data"], dtype=jnp.float32)
    else:
        f_obs, spatial_eff, conditionals = seed(sim_model, rng)(
            surrogate_decoder=None, y=None
        )
    return f_obs, spatial_eff, dict(zip(cond_names, conditionals))


def log_inference_run(
    hmc_res: tuple[dict, MCMC, dict],
    surrogate_conds: dict,
    results_dir: Path,
):
    samples, mcmc, post = hmc_res
    with open(results_dir / "hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / "hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    with open(results_dir / "hmc_summary.txt", "w") as out_file:
        out_file.write(capture_print_summary(mcmc))
    with open(results_dir / "mcmc.pkl", "wb") as out_file:
        pickle.dump(az.from_numpyro(mcmc), out_file)
    metrics = log_inference_metrics(hmc_res, surrogate_conds)
    print(metrics)
    with open(results_dir / "hmc_metrics.pkl", "wb") as out_file:
        pickle.dump(metrics, out_file)


def log_inference_metrics(hmc_res: tuple[dict, MCMC, dict], surrogate_conds: dict):
    samples, _, post = hmc_res
    metrics = {
        f"Real {name}": (val.squeeze().item() if val is not None else val)
        for name, val in surrogate_conds.items()
    }
    metrics.update(
        {
            f"Inferred {name}": samples[name].mean(axis=0).squeeze().item()
            for name in surrogate_conds
            if name in samples
        }
    )
    f_hat = post["obs"].mean(axis=0)
    f_obs = post["f"]
    obs_idx = post["obs_idxs"]
    unobs_idx = jnp.array([i for i in range(f_obs.shape[0]) if i not in obs_idx])

    def single_mse(x, y, name):
        metrics[f"{name}_MSE"] = jnp.mean((x - y) ** 2).item()
        if min(x.min(), y.mean()) >= 0:
            metrics[f"log_{name}_MSE"] = jnp.mean(
                (jnp.log(x + 1) - jnp.log(y + 1)) ** 2
            ).item()

    single_mse(f_hat, f_obs, "total")
    if len(obs_idx) < f_obs.shape[0]:
        single_mse(f_hat[obs_idx], f_obs[obs_idx], "observed")
        single_mse(f_hat[unobs_idx], f_obs[unobs_idx], "unobserved")
    wandb.log(metrics)
    return metrics


def capture_print_summary(mcmc):
    """Saves the mcmc table output into a str"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        mcmc.print_summary()
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout

    return output


def generate_obs_mask(rng: jax.Array, f_obs: jax.Array, infer_cfg: DictConfig):
    """Creates a mask which indicates to the inference model which locations to
    observe. If 'none's are present in the 'data' column they become unobserved,
    if 'num_obs_locations' argument is used then a subset of all the valid locations
    are randmoly chosen to be observed.

    Args:
        rng (jax.Array)
        f_obs (jax.Array): The sample to infer from
        infer_cfg (DictConfig): configuration for inference

    Returns:
        (Union[jax.Array, bool], jax.Array): boolean observe mask and
            indices of the observed locations
    """
    obs_idxs, obs_mask = jnp.arange(f_obs.shape[0]), True
    if jnp.isnan(f_obs).any():
        obs_mask = jnp.logical_not(jnp.isnan(f_obs))
        obs_idxs = jnp.where(obs_mask)[0]
    if infer_cfg.get("num_obs_locations") is not None:
        num_obs_locations = min(infer_cfg.num_obs_locations, len(obs_idxs))
        obs_idxs = jax.random.choice(rng, obs_idxs, (num_obs_locations,), replace=False)
        obs_mask = jnp.array([i in obs_idxs for i in range(f_obs.shape[0])])
    f_obs = f_obs.at[jnp.isnan(f_obs)].set(0)
    return obs_mask, obs_idxs


def init_priors(cfg: DictConfig):
    """inits the priors for inference and validates them"""
    priors = {}
    for pr, pr_dist in cfg.inference_model.priors.items():
        if pr_dist.numpyro_dist == "Delta":
            raise ValueError(
                "Cannot use the Delta distribution within the inference model, "
                f"please change the distribution of {pr}"
            )
        priors[pr] = instantiate(pr_dist)
    simulation_priors = priors.copy()
    # NOTE: Used to generate simulation data with specific values for experiments, could be partial
    # and is completed by priors' information
    if "simulation_priors" in cfg.inference_model:
        for pr, pr_dist in cfg.inference_model.simulation_priors.items():
            simulation_priors[pr] = instantiate(pr_dist)
    return priors, simulation_priors


def gen_inference_model(
    cfg: DictConfig,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    priors: Dict[str, dist.Distribution],
    surrogate_kwargs: dict = {},
):
    return instantiate(cfg.inference_model.model)(
        cfg,
        s,
        map_data,
        instantiate(cfg.inference_model.spatial_prior),
        priors,
        surrogate_kwargs,
    )


def load_ckpt(path: Union[str, Path]):
    "Load a checkpoint."
    if not isinstance(path, Path):
        path = Path(path)
    ckptr = PyTreeCheckpointer()
    ckpt = ckptr.restore(path.absolute())
    cfg = OmegaConf.create(ckpt["config"])
    model = instantiate(cfg.model)
    state = TrainState.create(
        apply_fn=model.apply,
        tx=optax.yogi(cosine_annealing_lr()),
        params=ckpt["state"]["params"],
        kwargs=ckpt["state"]["kwargs"],
    )
    return state, model


def get_results_dir(cfg, model_dir, obs_idxs, s):
    results_dir = model_dir / (
        f"{cfg.model.cls if cfg.inference_model.surrogate_model else 'Baseline_GP'}/"
        f"{cfg.inference_model.model.func}/"
    )
    if len(obs_idxs) < s.shape[0]:
        results_dir = results_dir / f"partial_obs_{len(obs_idxs)}"
    else:
        results_dir = results_dir / "complete_info"
    return results_dir


if __name__ == "__main__":
    main()
