#!/usr/bin/env python3
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Generator

import flax.linen as nn
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
from tqdm import tqdm
from utils.map_utils import gen_locations
from utils.obj_utils import build_model, generate_model_name, instantiate
from utils.plot_utils import log_vae_grid_plots, log_vae_map_plots

import wandb
from dl4bi.meta_learning.train_utils import cosine_annealing_lr, save_ckpt
from dl4bi.vae.train_utils import (
    Callback,
    TrainState,
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
        name=f"VAE_{cfg.exp_name}_{model_name}",
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    map_data, s = gen_locations(cfg.data)
    model = build_model(cfg.model, s)
    kwargs = {}
    if cfg.model.kwargs.decoder.cls == "FixedLocationTransfomer":
        kwargs = {"s": s}
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    # NOTE: large_batch_loader is used to compare the decoder distribution with true data
    spatial_prior = instantiate(cfg.inference_model.spatial_prior)
    priors = {
        pr: instantiate(pr_dist) for pr, pr_dist in cfg.inference_model.priors.items()
    }
    train_loader, test_loader, large_batch_loader, cond_names = (
        build_spatial_dataloaders(
            rng,
            cfg,
            map_data,
            s,
            priors,
            spatial_prior,
        )
    )
    valid_step = get_valid_step(cfg.model, cond_names)
    decoder_only = cfg.model.cls == "DeepRV"
    z_dim = s.shape[0] if decoder_only else model.z_dim
    callback_fn = log_vae_grid_plots if map_data is None else log_vae_map_plots
    state = train(
        rng_train,
        model,
        optimizer,
        get_train_step(cfg.model, cond_names),
        valid_step,
        train_loader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        log_every_n=100,
        callbacks=[
            Callback(
                callback_fn(
                    map_data,
                    s,
                    cond_names,
                    z_dim,
                    large_batch_loader,
                    decoder_only,
                ),
                cfg.plot_interval,
            )
        ],
        **kwargs,
    )
    validate(
        rng_test,
        state,
        valid_step,
        test_loader,
        cfg.valid_num_steps,
        name="Test",
        **kwargs,
    )
    results_path = Path(
        f"results/{cfg.exp_name}/{spatial_prior.__name__}/{cfg.seed}/{model_name}"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, results_path.with_suffix(".ckpt"))


def build_spatial_dataloaders(
    rng: Array,
    cfg: DictConfig,
    map_data: gpd.GeoDataFrame,
    s: Array,
    priors: dict[str, dist.Distribution],
    spatial_prior: Callable,
):
    """Generates the GP dataloader for training or inference for
    a specific distance based or graph model based kernel

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
    rng_train, rng_test, rng_large_batch = random.split(rng, 3)
    spatial_model, cond_names = gen_saptial_prior(
        cfg, s, spatial_prior, priors, map_data
    )

    def dataloader(rng_data, bs=cfg.data.batch_size):
        seeded_model = seed(spatial_model, rng_data)
        while True:
            yield seeded_model(surrogate_decoder=None, batch_size=bs)

    return (
        dataloader(rng_train),
        dataloader(rng_test),
        dataloader(rng_large_batch, bs=cfg.data.large_batch_size),
        cond_names,
    )


def train(
    rng: Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    train_step: Callable,
    valid_step: Callable,
    loader: Generator,
    train_num_steps: int = 100000,
    valid_num_steps: int = 25000,
    valid_interval: int = 25000,
    log_every_n: int = 100,
    callbacks: list[Callback] = [],
    **kwargs,
):
    rng_params, rng_extra, rng_train = random.split(rng, 3)
    f, z, conditionals = next(loader)
    rngs = {"params": rng_params, "extra": rng_extra}
    x = z if model.__class__.__name__ == "DeepRV" else f
    m_kwargs = model.init(rngs, x, conditionals, **kwargs)
    params = m_kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(x, conditionals, **kwargs)
    state = TrainState.create(
        apply_fn=model.apply, params=params, kwargs=m_kwargs, tx=optimizer
    )
    print(param_count)

    losses = np.zeros((train_num_steps,))
    for i in (pbar := tqdm(range(train_num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng_train = random.split(rng_train)
        batch = next(loader)
        state, losses[i] = train_step(rng_step, state, batch, **kwargs)
        if (i + 1) % log_every_n == 0:
            avg = jnp.mean(losses[i - log_every_n : i])
            pbar.set_postfix(loss=f"{avg:.3f}")
            wandb.log({"loss": avg})
        if (i + 1) % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            validate(
                rng_valid,
                state,
                valid_step,
                loader,
                valid_num_steps,
                **kwargs,
            )
        for cbk in callbacks:
            if (i + 1) % cbk.interval == 0:
                cbk.fn(i, rng_step, state, model, loader, **kwargs)
    return state


def validate(
    rng: Array,
    state: TrainState,
    valid_step: Callable,
    loader: Generator,
    valid_num_steps: int = 5000,
    name: str = "Validation",
    **kwargs,
):
    metrics = defaultdict(list)
    for _ in (_ := tqdm(range(valid_num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng = random.split(rng)
        batch = next(loader)
        m = valid_step(rng_step, state, batch, prefix=name, **kwargs)
        for k, v in m.items():
            if v is not None:
                metrics[k] += [v]
    if "ls" in metrics:
        ls = jnp.array(metrics["ls"])
        norm_mse = jnp.array(metrics[f"{name} norm MSE"])
        for ls_r in [[0, 5], [5, 10], [10, 20], [20, 50]]:
            ls_range_name = f"{name} ls {ls_r[0]}-{ls_r[1]}"
            low_ls = jnp.logical_and(ls_r[0] < ls, ls < ls_r[1])
            metrics[f"{ls_range_name} norm MSE"] = norm_mse[low_ls]
        del metrics["ls"]
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    wandb.log(metrics)
    print(metrics)


def get_train_step(model_cfg: DictConfig, cond_names: list[str]):
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    train_step = elbo_train_step
    if model_cfg.cls == "DeepRV":
        train_step = partial(deep_RV_train_step, var_idx=var_idx)
    elif model_cfg.cls == "PriorCVAE":
        train_step = prior_cvae_train_step
    return train_step


def get_valid_step(model_cfg: DictConfig, cond_names: list[str]):
    ls_idx = None if "ls" not in cond_names else cond_names.index("ls")
    var_idx = None if "var" not in cond_names else cond_names.index("var")
    model_name = model_cfg.cls
    decoder_only = model_name == "DeepRV"

    def valid_step(rng, state, batch, prefix: str = "", **kwargs):
        f, z, conditionals = batch
        var = 1 if var_idx is None else conditionals[var_idx].squeeze()
        ls = None if ls_idx is None else conditionals[ls_idx].squeeze()
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng}
        z_mu, z_std = None, None
        f_hat = jit(state.apply_fn)(
            params, z if decoder_only else f, conditionals, **kwargs, rngs=rngs
        )
        if not decoder_only:
            f_hat, z_mu, z_std = f_hat
        mse_score = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        # NOTE: Normalizing the mse score with variance, aN(0,K)~N(0, a^2K), and
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

        return {
            f"{prefix} loss": loss,
            f"{prefix} norm MSE": norm_mse_score,
            "ls": ls if ls is not None else None,
        }

    return valid_step


if __name__ == "__main__":
    main()
