"""This script was used to compare several Approximated inference methods
Vs DeepRV. We weree unable to get reasonable results here and therefore
didn't report these in the paper."""

from pathlib import Path
from typing import Callable, Union

import hydra
import jax
import jax.numpy as jnp
from infer import (
    gen_inference_model,
    gen_obs_data,
    generate_obs_mask,
    get_results_dir,
    init_priors,
    log_inference_run,
)
from jax import random
from numpyro.infer import SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoLaplaceApproximation
from numpyro.optim import Adam
from omegaconf import DictConfig, OmegaConf
from utils.map_utils import gen_locations
from utils.plot_utils import plot_inference_run

import wandb


@hydra.main("configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    spatial_prior = cfg.inference_model.spatial_prior
    # TODO(jhonathan/danj): replace initialization of guide
    guide = AutoDiagonalNormal
    model_name = f"{guide.__name__}_{spatial_prior.func}"
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=f"Approx_infer_{cfg.exp_name}_{model_name}_{cfg.inference_model.model.func}",
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    rng, rng_idxs, rng_infer, rng_plot = random.split(random.key(cfg.seed), 4)
    map_data, s = gen_locations(cfg.data)
    model_dir = Path(f"results/{cfg.exp_name}/{spatial_prior.func}/{cfg.seed}")
    priors, simulation_priors = init_priors(cfg)
    # NOTE: the inference model expects to have this exact API
    infer_model, cond_names = gen_inference_model(cfg, s, map_data, priors)
    # NOTE: Generates a model for creating the simulated data
    sim_model, _ = gen_inference_model(cfg, s, map_data, simulation_priors)
    f_obs, spatial_eff, conds = gen_obs_data(rng, sim_model, map_data, cond_names)
    obs_mask, obs_idxs = generate_obs_mask(rng_idxs, f_obs, cfg.inference_model)
    samples, post = approx_infer(rng_infer, guide, infer_model, f_obs, obs_mask)
    post.update(
        {
            "s": s,
            "f": f_obs,
            "obs_idxs": obs_idxs,
            "spatial_eff": spatial_eff,
            **conds,
        }
    )
    results_dir = get_results_dir(
        cfg.inference_model.model.func,
        model_dir,
        obs_idxs,
        s,
        model_name,
        approx_infer=True,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    log_inference_run((samples, None, post), conds, results_dir)
    plot_inference_run(
        rng_plot,
        cfg,
        model_name,
        (samples, None, post),
        f_obs,
        conds,
        priors,
        map_data,
    )


def approx_infer(
    rng: jax.Array,
    guide,
    model: Callable,
    f_obs: jax.Array,
    obs_mask: Union[jax.Array, bool],
    num_iterations: int = 1000,
):
    """runs approximate inference on given inference model and observed f"""
    rng_init, rng_samples, rng_post = jax.random.split(rng, 3)
    optimizer = Adam(step_size=0.01)
    if guide == AutoLaplaceApproximation:
        guide = guide(
            model,
            hessian_fn=lambda f, x: jax.hessian(f)(x) + 1e-3 * jnp.eye(x.shape[0]),
        )
    else:
        guide = guide(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    init_state = svi.init(rng_init, y=f_obs, obs_mask=obs_mask)
    state, _ = jax.lax.scan(
        lambda state, _: svi.update(state, y=f_obs),
        init_state,
        jnp.arange(num_iterations),
    )
    params = svi.get_params(state)
    samples = guide.sample_posterior(rng_samples, params, sample_shape=(10000,))
    post = Predictive(model, samples)(rng_post)
    return samples, post


if __name__ == "__main__":
    main()
