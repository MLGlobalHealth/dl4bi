import pickle

#!/usr/bin/env python3
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import hydra
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import optax
from jax import random
from map_utils import get_raw_map_data, process_map
from models import PriorCVAE
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint import PyTreeCheckpointer
from plot_utils import (
    plot_covariance,
    plot_histograms,
    plot_infer_observed_coverage,
    plot_infer_realizations,
    plot_trace,
    plot_violin,
)
from sps.kernels import matern_1_2, matern_3_2, matern_5_2, periodic, rbf
from sps.priors import Prior
from vae import gp_dataloaders, graph_dataloaders
from vae_train_utils import generate_model_name, get_model_kwargs, instantiate

import wandb
from dl4bi.meta_regression.train_utils import (
    TrainState,
    cosine_annealing_lr,
)


@hydra.main("configs", config_name="default_infer", version_base=None)
def main(cfg: DictConfig):
    is_gp = cfg.is_gp
    decoder_only = cfg.model.kwargs.get("decoder_only", False)
    model_name = generate_model_name(cfg, is_gp, decoder_only)
    if cfg.infer.run_gp_baseline:
        model_name = "Baseline_GP"
    run_name = cfg.get("name", f"Infer_{model_name}")
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,
    )
    rng, rng_plot, idxs_rng = random.split(random.key(cfg.seed), 3)
    s, _, _ = process_map(cfg.data)
    results_dir = Path(
        f"results/{cfg.project}/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}"
    )
    map_data = get_raw_map_data(cfg.data.name)
    infer_loader, conditionals_names = infer_model_dataloader(
        rng, cfg, s, map_data, is_gp
    )
    f_batch, conditionals = next(iter(infer_loader))
    conditionals = dict(zip(conditionals_names, conditionals))
    # NOTE: obs_idxs taking all LTAs as samples, it's possible to give partial information also
    obs_idxs, obs_mask = jnp.arange(len(map_data)), True
    if cfg.infer.get("num_context_points") is not None:
        obs_idxs = jax.random.choice(
            idxs_rng, obs_idxs, (cfg.infer.num_context_points,), replace=False
        )
        obs_mask = jnp.array([i in obs_idxs for i in range(len(s))])
    priors = {pr: instantiate(pr_dist) for pr, pr_dist in cfg.infer.priors.items()}
    kernel = instantiate(cfg.kernel.kwargs.kernel)
    z_to_y_hat_fn, z_dim, vae_kwargs = None, None, {}
    if not cfg.infer.run_gp_baseline:
        state, _ = load_ckpt((results_dir / model_name).with_suffix(".ckpt"))
        vae_kwargs = get_model_kwargs(
            s, cfg.data.pre_process, map_data, cfg.data.graph_construction
        )
        z_to_y_hat_fn = decoder_z_to_y_hat(state, **vae_kwargs)
        z_dim = cfg.model.kwargs.z_dim
    inference_model = build_inference_model(
        s, obs_mask, kernel, priors, z_to_y_hat_fn, z_dim
    )
    f = f_batch[0].squeeze()
    hmc_res = _hmc(cfg, inference_model, s, f, conditionals, obs_idxs)
    log_inference_run(
        rng_plot,
        hmc_res,
        s,
        f_batch,
        conditionals,
        kernel,
        priors,
        map_data,
        model_name,
        results_dir,
    )


def _hmc(
    cfg: DictConfig,
    model,
    s: jax.Array,
    f: jax.Array,
    conditionals: dict,
    obs_idxs: jax.Array,
):
    """runs HMC on given model and observed f"""
    nuts = NUTS(model, init_strategy=init_to_median(num_samples=10))
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    mcmc = MCMC(nuts, **cfg.infer.mcmc)
    mcmc.run(k1, f.squeeze())
    mcmc.print_summary()
    samples = mcmc.get_samples()
    post = Predictive(model, samples)(k2)
    post.update(
        {
            "s": s,
            "f": f,
            "obs_idxs": obs_idxs,
            **conditionals,
        }
    )
    return samples, mcmc, post


def infer_model_dataloader(
    rng: jax.Array,
    cfg: DictConfig,
    s: jax.Array,
    map_data: gpd.GeoDataFrame,
    is_gp: bool,
):
    """Generates the dataloader for inference model. Wraps GP
    or graph based models with additional complexities like
    Poisson.

    Args:
        rng (jax.Array)
        cfg (DictConfig): config dict
        s (jax.Array): locations
        map_data (gpd.GeoDataFrame): original geopandas
        is_gp (bool): flags whether to wrap GPs or graph based models

    Returns:
        the inference dataloader
    """
    rng_beta, rng_loader = random.split(rng)
    beta = random.normal(rng_beta, shape=(s.shape[0], 1))
    beta = jnp.repeat(beta[None, ...], cfg.batch_size, axis=0)
    if is_gp:
        loader, _, _, cond_names = gp_dataloaders(
            rng_loader,
            cfg,
            s,
        )
    else:
        loader, _, _, cond_names = graph_dataloaders(
            rng_loader,
            cfg,
            map_data,
        )

    def infer_loader(rng_infer):
        while True:
            rng_poisson, rng_infer = random.split(rng_infer)
            mu_batch, _, conditionals = next(loader)
            lambda_ = jnp.exp(mu_batch + beta)
            yield random.poisson(rng_poisson, lambda_), conditionals

    return infer_loader(rng_loader), cond_names


def build_inference_model(
    s: jax.Array,
    obs_mask: Union[bool, jax.Array],
    kernel,
    priors: dict,
    z_to_y_hat_fn=None,
    z_dim: Optional[int] = None,
    jitter: float = 1e-4,
):
    """
    Builds an inference model for both GP baseline and surrogate inference.

    Args:
        s: Locations (n_locations, dim_s).
        obs_mask: Mask for observed data.
        kernel: Kernel function.
        priors: Dictionary of prior distributions.
        state: (Optional) State for the surrogate model.
        surrogate_z_dim: (Optional) Dimensionality of the latent space (z).
        z_to_y_hat_fn: (Optional) Function to map z to predictions (surrogate only).
        jitter: Small jitter value for numerical stability.
        vae_kwargs: (Optional) Additional arguments for surrogate decoding.

    Returns:
        A NumPyro model function.
    """
    n_locations = s.shape[0]
    I_jitter = jitter * jnp.eye(n_locations)
    if z_to_y_hat_fn is not None:
        z_dist = dist.Normal()

    def inference_model(y=None):
        variance = numpyro.sample("var", priors["var"]).squeeze()
        lengthscale = numpyro.sample("ls", priors["ls"]).squeeze()
        conditionals = [jnp.array([variance]), jnp.array([lengthscale])]
        if kernel.__name__ == "periodic":
            conditionals += [numpyro.sample("period", priors["period"])]
        if z_to_y_hat_fn is None:
            K = kernel(s, s, *conditionals) + I_jitter
            mu = numpyro.sample("mu", dist.MultivariateNormal(0, K))
        else:
            z = numpyro.sample("z", z_dist, sample_shape=(1, z_dim))
            mu = numpyro.deterministic("mu", z_to_y_hat_fn(z, conditionals).squeeze())
        beta = numpyro.sample("beta", priors["beta"], sample_shape=(n_locations,))
        lambda_ = jnp.exp(beta + mu)
        with numpyro.handlers.mask(mask=obs_mask):
            numpyro.sample("obs", dist.Poisson(rate=lambda_), obs=y)

    return inference_model


def decoder_z_to_y_hat(state: TrainState, **vae_kwargs):
    """Wraps a VAE model to issue decoder only calls within numpyro model

    Args:
        state (TrainState): state to wrap

    Returns: the decoding function
    """

    def z_to_y_hat_fn(z, conditionals):
        return state.apply_fn(
            {"params": state.params, **state.kwargs},
            z[..., None],
            conditionals,
            decode_only=True,
            **vae_kwargs,
        )[0]

    return z_to_y_hat_fn


def log_inference_run(
    rng_plot: jax.Array,
    hmc_res: tuple[dict, MCMC, dict],
    s: jax.Array,
    f_batch: jax.Array,
    conditionals: dict,
    kernel,
    priors: dict,
    map_data: gpd.GeoDataFrame,
    model_name: str,
    results_dir: Path,
):
    samples, mcmc, post = hmc_res
    with open(results_dir / f"{model_name}_hmc_samples.pkl", "wb") as out_file:
        pickle.dump(samples, out_file)
    with open(results_dir / f"{model_name}_hmc_pp.pkl", "wb") as out_file:
        pickle.dump(post, out_file)
    plot_infer_observed_coverage(post, map_data)
    plot_infer_realizations(rng_plot, map_data, f_batch, post)
    plot_trace(samples, mcmc, conditionals, model_name)
    plot_covariance(samples, conditionals, model_name, kernel, s)
    plot_histograms(samples, conditionals, model_name, priors)
    plot_violin(post, f_batch, model_name, log_scale=True)


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
    return state, cfg


if __name__ == "__main__":
    main()
