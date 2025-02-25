import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Union

import flax.linen as nn
import geopandas as gpd
import hydra
import jax
import jax.numpy as jnp
from infer import gen_inference_model, load_ckpt
from jax import random
from numpyro.handlers import seed
from omegaconf import DictConfig, OmegaConf
from scipy.optimize import minimize
from utils.map_utils import generate_adjacency_matrix, process_map
from utils.plot_utils import plot_EB_scatter_conditionals
from utils.obj_utils import generate_model_name, instantiate

import wandb
from dl4bi.vae.train_utils import TrainState, generate_surrogate_decoder


@hydra.main("configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # NOTE: the model name has to match the model name used in inference
    model_name = generate_model_name(cfg)
    spatial_prior = instantiate(cfg.inference_model.spatial_prior)
    state = None
    model_dir = Path(f"results/{cfg.exp_name}/{spatial_prior.__name__}/{cfg.seed}")
    if not cfg.inference_model.surrogate_model:
        model_name = f"Baseline_GP_{spatial_prior.__name__}"
    else:
        state, dec_model = load_ckpt((model_dir / model_name).with_suffix(".ckpt"))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=f"Infer_Empirical_Bayes_{cfg.exp_name}_{model_name}",
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    map_data = gpd.read_file(cfg.data.map_path)
    s = process_map(map_data)
    priors = {
        pr: instantiate(pr_dist) for pr, pr_dist in cfg.inference_model.priors.items()
    }
    surrogate_kwargs = {}
    if cfg.model.kwargs.decoder.cls == "FixedLocationTransfomer":
        surrogate_kwargs = {"s": s}
    inference_model, conds_names = gen_inference_model(
        cfg, s, map_data, priors, surrogate_kwargs
    )
    loader = gen_obs_data(rng, inference_model, dec_model, state)
    adj_mat = None
    if spatial_prior.__name__ in ["car", "icar", "bym"]:
        adj_mat = generate_adjacency_matrix(map_data, cfg.data.graph_construction)
    num_samples = cfg.emprical_bayes_samples
    real_conds = jnp.zeros((num_samples, len(conds_names)))
    inferred_conds = jnp.zeros_like(real_conds)
    for i in range(num_samples):
        f_obs, conds = next(loader)
        infer_cond = empirical_infer(s, f_obs, conds, spatial_prior, adj_mat)
        real_conds = real_conds.at[i].set(jnp.array(conds).squeeze())
        inferred_conds = inferred_conds.at[i].set(jnp.array(infer_cond).squeeze())
    results_dir = (
        model_dir
        / f"{model_name.replace(f'_{spatial_prior.__name__}', '')}/Empirical_Bayes"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    log_EB_run(real_conds, inferred_conds, conds_names, results_dir)


def gen_obs_data(
    rng: jax.Array,
    model: Callable,
    dec_model: nn.Module,
    state: Union[TrainState, None],
):
    """Generates loader for observed data by sampling the model's decoder
    or sampling an actual spatial model.

    Args:
        rng (jax.Array)
        model (Callable): inference model to generate the data from
        state (TrainState): the model's state to decode from

    Returns:
        the observed data dataloader
    """
    surrogate_decoder = None
    if state is not None:
        surrogate_decoder = generate_surrogate_decoder(state, dec_model)

    def obs_loader(rng):
        seeded_model = seed(model, rng)
        while True:
            sample, _, conditionals = seeded_model(surrogate_decoder=surrogate_decoder)
            yield sample, conditionals

    return obs_loader(rng)


def marginal_log_likelihood(
    conditionals: list,
    s: jax.Array,
    f_obs: jax.Array,
    spatial_prior: Callable,
    adj_mat: Union[jax.Array, None],
):
    """Computes the marginal likelihood of a gaussian given the spatial
    prior's appropriate conditionals.

    Args:
        conditionals (list): conditionals for prior and noise
        s (jax.Array): locations
        f_obs (jax.Array): observed data
        spatial_prior (Union[jax.Array, None]): GP kernel func or placeholder func
        adj_mat (jax.Array): Adjacency matrix in case graph model is used

    Raises:
        ValueError: If an unsupported kernel is used

    Returns:
        float: marginal likelihood of a gaussian
    """
    spatial_conds, noise_std = conditionals[:-1], conditionals[-1]
    prior_name = spatial_prior.__name__
    n = s.shape[0]
    if prior_name in ["rbf", "periodic", "matern_3_2", "matern_5_2", "matern_1_2"]:
        K = spatial_prior(s, s, *spatial_conds)
    elif prior_name in ["car"]:
        D = jnp.diag(adj_mat.sum(axis=1))
        tau = spatial_conds[0].squeeze()
        alpha = spatial_conds[1].squeeze()
        precision_matrix = D - (alpha * adj_mat)
        K = (1 / tau) * jnp.linalg.pinv(precision_matrix, hermitian=True)
    else:
        raise ValueError(
            f"Empirical Bayes doesn't support the chosen prior {prior_name}"
        )
    K = K + jnp.eye(n) * noise_std**2
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, f_obs))
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diagonal(L)))
    # NOTE: Negative marginal log likelihood
    mll = 0.5 * jnp.dot(f_obs, alpha) + 0.5 * log_det - 0.5 * n * jnp.log(2 * jnp.pi)
    return mll


def empirical_infer(
    s: jax.Array,
    f_obs: jax.Array,
    initial_conditionals: list,
    spatial_prior: Callable,
    adj_mat: Union[jax.Array, None],
):
    """Finds point-wise optimal conditional parameters to fit the generated f_obs

    Args:
        s (jax.Array): locations
        f_obs (jax.Array): observed data
        initial_conditionals (list): initial conditionals for prior and noise
        spatial_prior (Callable): GP kernel func or placeholder func
        adj_mat (Union[jax.Array, None]): Adjacency matrix in case graph model is used

    Returns:
        list of best fitted parameters
    """
    # NOTE: bounds for conditionals, noise is unbounded
    bounds = [(0.05, 1)] * (len(initial_conditionals) - 1) + [(1e-5, None)]
    result = minimize(
        partial(
            marginal_log_likelihood,
            s=s,
            f_obs=f_obs,
            spatial_prior=spatial_prior,
            adj_mat=adj_mat,
        ),
        x0=initial_conditionals,
        bounds=bounds,
    )
    return result.x


def log_EB_run(
    real_conds: jax.Array,
    inferred_conds: jax.Array,
    cond_names: list[str],
    results_dir: Path,
):
    plot_path = plot_EB_scatter_conditionals(real_conds, inferred_conds, cond_names)
    wandb.log({"Scatter plot": wandb.Image(plot_path)})
    metric_dict = {}
    data_dict = {}
    for i, c_name in enumerate(cond_names):
        real_c, inf_c = real_conds[:, i], inferred_conds[:, i]
        mse_cond = jnp.mean((real_c - inf_c) ** 2).item()
        metric_dict[f"{c_name} MSE"] = mse_cond
        data_dict[f"real_{c_name}"] = real_c
        data_dict[f"inf_{c_name}"] = inf_c
    print(metric_dict)
    with open(results_dir / "cond_metrics.pkl", "wb") as out_file:
        pickle.dump(metric_dict, out_file)
    with open(results_dir / "cond_data.pkl", "wb") as out_file:
        pickle.dump(data_dict, out_file)
    wandb.log(metric_dict)


if __name__ == "__main__":
    main()
