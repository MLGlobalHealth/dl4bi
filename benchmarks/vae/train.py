from functools import partial
from pathlib import Path
from typing import Callable

import geopandas as gpd
import hydra
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
from inference_models.inference_models import gen_saptial_prior
from jax import Array, jit, random
from jax.scipy.stats import norm
from numpyro.handlers import seed
from omegaconf import DictConfig, OmegaConf
from utils.map_utils import gen_locations
from utils.obj_utils import build_model, generate_model_name, instantiate
from utils.plot_utils import log_vae_grid_plots, log_vae_map_plots

import wandb
from dl4bi.core.train import Callback, cosine_annealing_lr, evaluate, save_ckpt, train
from dl4bi.vae.train_utils import (
    deep_RV_train_step,
    elbo_train_step,
    prior_cvae_train_step,
)


@hydra.main("configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # NOTE: the model name has to match the model name used in inference
    model_name = generate_model_name(cfg)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=f"VAE_{cfg.exp_name}_{model_name}_{cfg.inference_model.spatial_prior.func}",
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test, rng_lb, rng_plot = random.split(rng, 4)
    map_data, s = gen_locations(cfg.data)
    model = build_model(cfg.model, s)
    optimizer = get_optimizer(cfg)
    spatial_prior = instantiate(cfg.inference_model.spatial_prior)
    priors = {
        pr: instantiate(pr_dist) for pr, pr_dist in cfg.inference_model.priors.items()
    }
    # NOTE: large_batch_loader is used to compare the decoder distribution with true data
    loader_gen, cond_names = build_spatial_dataloaders(
        cfg, map_data, s, priors, spatial_prior
    )
    valid_step = gen_valid_step(cfg.model, cond_names)
    decoder_only = cfg.model.cls == "DeepRV"
    z_dim = s.shape[0] if decoder_only else model.z_dim
    callback_fn = log_vae_grid_plots if map_data is None else log_vae_map_plots
    state = train(
        rng_train,
        model,
        optimizer,
        gen_train_step(cfg.model, cond_names),
        cfg.train_num_steps,
        loader_gen,
        valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        loader_gen,
        valid_monitor_metric="loss",
        callbacks=[
            Callback(
                callback_fn(
                    map_data,
                    s,
                    cond_names,
                    z_dim,
                    loader_gen(rng_plot),
                    loader_gen(rng_lb, cfg.data.large_batch_size),
                    decoder_only,
                    cfg.data,
                    model,
                ),
                cfg.plot_interval,
            )
        ],
    )
    log_run(evaluate(rng_test, state, valid_step, loader_gen, cfg.valid_num_steps))
    results_path = Path(
        f"results/{cfg.exp_name}/{spatial_prior.__name__}/{cfg.seed}/{model_name}"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, results_path.with_suffix(".ckpt"))


def build_spatial_dataloaders(
    cfg: DictConfig,
    map_data: gpd.GeoDataFrame,
    s: Array,
    priors: dict[str, dist.Distribution],
    spatial_prior: Callable,
):
    """Generates the spatial prior dataloader for training for
    a specific distance based GP or graph model based kernel

    Args:
        rng (Array)
        cfg (DictConfig): Run configuration
        map_data (gpd.GeoDataFrame): map data to construct the graph from (graph model case)
        s (Array): locations on map
        priors (dict[str, dist.Distribution]): hyperparameter priors for sampling
        spatial_prior (Callable): either gp kernel function, or spatial prior name

    Returns:
        train loader, test loader, large batch loader, and surrogates models' conditionals names
    """
    spatial_model, cond_names = gen_saptial_prior(
        cfg, s, spatial_prior, priors, map_data
    )

    def dataloader(rng_data, bs=cfg.data.batch_size):
        while True:
            rng_data, _ = random.split(rng_data)
            seeded_model = seed(spatial_model, rng_data)
            f, z, conditionals = seeded_model(surrogate_decoder=None, batch_size=bs)
            yield {"s": s, "f": f, "z": z, "conditionals": conditionals}

    return dataloader, cond_names


def log_run(metrics: dict):
    """log train run"""
    if "ls" in metrics:
        ls = jnp.array(metrics["ls"])
        norm_mse = jnp.array(metrics["norm MSE"])
        for ls_r in [[0, 5], [5, 10], [10, 20], [20, 50]]:
            ls_range_name = f"ls {ls_r[0]}-{ls_r[1]}"
            low_ls = jnp.logical_and(ls_r[0] < ls, ls < ls_r[1])
            metrics[f"{ls_range_name} norm MSE"] = norm_mse[low_ls]
        del metrics["ls"]
    metrics = {f"Test {k}": np.mean(v) for k, v in metrics.items()}
    wandb.log(metrics)
    print(metrics)


def gen_train_step(model_cfg: DictConfig, cond_names: list[str]):
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    train_step = elbo_train_step
    if model_cfg.cls == "DeepRV":
        train_step = partial(deep_RV_train_step, var_idx=var_idx)
    elif model_cfg.cls == "PriorCVAE":
        train_step = prior_cvae_train_step
    return train_step


def gen_valid_step(model_cfg: DictConfig, cond_names: list[str]):
    ls_idx = None if "ls" not in cond_names else cond_names.index("ls")
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    model_name = model_cfg.cls
    decoder_only = model_name == "DeepRV"

    def valid_step(rng, state, batch):
        f, conditionals = batch["f"], batch["conditionals"]
        var = 1 if var_idx is None else conditionals[var_idx].squeeze()
        ls = None if ls_idx is None else conditionals[ls_idx].squeeze()
        z_mu, z_std = None, None
        f_hat = jit(state.apply_fn)(
            {"params": state.params, **state.kwargs}, **batch, rngs={"extra": rng}
        )
        if not decoder_only:
            f_hat, z_mu, z_std = f_hat
        mse_score = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        # NOTE: Normalizing the mse score with variance, aN(0,K)~N(0, a^2K)
        norm_mse_score = (1 / var) * mse_score
        loss = norm_mse_score
        if not decoder_only:
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = (
                (1 / (2 * 0.9)) * mse_score
                if model_name == "PriorCVAE"
                else -norm.logpdf(f, f_hat, 1.0).mean()
            )
            loss = logp + kl_div.mean()
        return {"loss": loss, "norm MSE": norm_mse_score, "ls": ls}

    return valid_step


def get_optimizer(cfg: DictConfig):
    if cfg.cosine_annealing:
        lr_schedule = cosine_annealing_lr(
            cfg.train_num_steps, cfg.lr_peak, cfg.lr_pct_warmup
        )
        return optax.chain(
            optax.clip_by_global_norm(cfg.clip_max_norm), optax.yogi(lr_schedule)
        )
    else:
        return optax.yogi(cfg.lr_peak)


if __name__ == "__main__":
    main()
