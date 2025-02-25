#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Generator

import flax.linen as nn
import geopandas as gpd
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optax
from inference_models.inference_models import gen_saptial_prior
from jax import random
from numpyro.handlers import seed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from utils.map_utils import process_map
from utils.plot_utils import log_vae_map_plots
from utils.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    generate_model_name,
    get_train_step,
    get_valid_step,
    instantiate,
    save_ckpt,
)

import wandb


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
    map_data = gpd.read_file(cfg.data.map_path)
    s = process_map(map_data)
    model = instantiate(cfg.model)
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
                log_vae_map_plots(
                    map_data,
                    s,
                    cond_names,
                    s.shape[0],
                    large_batch_loader,
                    cfg.model.cls == "DeepRV",
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
    rng: jax.Array,
    cfg: DictConfig,
    map_data: gpd.GeoDataFrame,
    s: jax.Array,
    priors: dict[str, dist.Distribution],
    spatial_prior: Callable,
):
    """Generates the GP dataloader for training or inference for
    a specific distance based or graph model based kernel

    Args:
        rng (jax.Array)
        cfg (DictConfig): Run configuration
        map_data (gpd.GeoDataFrame): map data to construct the graph from (graph model case)
        s (jax.Array): locations on map
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
    rng: jax.Array,
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
    rng: jax.Array,
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


if __name__ == "__main__":
    main()
