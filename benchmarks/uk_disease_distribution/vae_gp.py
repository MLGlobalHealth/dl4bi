#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Optional, Union

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random
from jax.scipy.stats import norm
from map_utils import get_raw_map_data, process_map
from omegaconf import DictConfig, OmegaConf
from plot_utils import log_vae_map_plots
from prior_cvae import PriorCVAE
from sps.gp import GP
from sps.kernels import matern_3_2, periodic, rbf
from sps.priors import Prior
from tqdm import tqdm
from vae_train_utils import elbo_train_step, mse_train_step

import wandb
from dl4bi.core import *  # noqa: F403
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    save_ckpt,
)
from dl4bi.vae import DeepChol, train_utils


@hydra.main("configs", config_name="default_vae", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", f"VAE_GP_{cfg.model.cls}_{cfg.data.sampling_policy}")
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))

    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    map_data = get_raw_map_data(cfg.data.name)
    s, _, _ = process_map(cfg.data)
    model, gp = instantiate(cfg.model), instantiate(cfg.kernel)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )

    state = train(
        rng_train,
        gp,
        s,
        model,
        optimizer,
        isinstance(model, (DeepChol,)),
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        cfg.batch_size,
        callbacks=[
            Callback(
                log_vae_map_plots(map_data, s, ["var", "ls"], cfg.model.kwargs.z_dim),
                cfg.plot_interval,
            )
        ],
    )
    results = validate(
        rng_test,
        gp,
        s,
        state,
        isinstance(model, (DeepChol,)),
        cfg.valid_num_steps,
        cfg.batch_size,
        log_results=True,
    )
    path = Path(
        f"results/uk_disease_dist/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".pkl"), "wb") as save_file:
        pickle.dump(results, save_file)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def dataloader(
    rng: jax.Array, gp: GP, s: jax.Array, batch_size: int = 32, approx: bool = False
):
    while True:
        _, rng = random.split(rng)
        f, var, ls, _, z = gp.simulate(rng, s, batch_size, approx)
        yield f, z, [var, ls]


def train(
    rng: jax.Array,
    gp: GP,
    s: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    is_decoder_only: bool,
    train_num_steps: int = 100000,
    valid_num_steps: int = 25000,
    valid_interval: int = 25000,
    batch_size: int = 32,
    log_every_n: int = 100,
    callbacks: list[Callback] = [],
    state: Optional[TrainState] = None,
):
    rng_data, rng_params, rng_extra, rng_train = random.split(rng, 4)
    loader = dataloader(rng_data, gp, s, batch_size)
    f, z, conditionals = next(loader)
    rngs = {"params": rng_params, "extra": rng_extra}

    x = z if is_decoder_only else f
    train_step = mse_train_step if is_decoder_only else elbo_train_step
    kwargs = model.init(rngs, x, conditionals)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(x, conditionals)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params if state is None else state.params,
        kwargs=kwargs if state is None else state.kwargs,
        tx=optimizer,
    )
    print(param_count)

    losses = np.zeros((train_num_steps,))
    for i in (pbar := tqdm(range(train_num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng_train = random.split(rng_train)
        batch = next(loader)
        state, losses[i] = train_step(rng_step, state, batch)
        if (i + 1) % log_every_n == 0:
            avg = jnp.mean(losses[i - log_every_n : i])
            pbar.set_postfix(loss=f"{avg:.3f}")
            wandb.log({"loss": avg})
        if (i + 1) % valid_interval == 0:
            rng_valid, rng_train = random.split(rng_train)
            validate(
                rng_valid,
                gp,
                s,
                state,
                is_decoder_only,
                valid_num_steps,
                batch_size,
            )
        for cbk in callbacks:
            if (i + 1) % cbk.interval == 0:
                cbk.fn(i, rng_step, state, loader, model)
    return state


def validate(
    rng: jax.Array,
    gp: GP,
    s: jax.Array,
    state: train_utils.TrainState,
    is_decoder_only: bool,
    valid_num_steps: int = 5000,
    batch_size: int = 32,
    log_results: bool = False,
):
    rng_data, rng_extra = random.split(rng)
    loader = dataloader(rng_data, gp, s, batch_size)
    losses = np.zeros((valid_num_steps,))
    results = []
    for i in (_ := tqdm(range(valid_num_steps), unit="batch", dynamic_ncols=True)):
        batch = next(loader)
        f, z, conditionals = batch
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng_extra}
        if is_decoder_only:
            f_hat = jit(state.apply_fn)(params, z, conditionals)
            losses[i] = optax.squared_error(f_hat, f.squeeze()).mean()
        else:
            # NOTE: ELBO, assumes normal gaussian
            f_hat, z_mu, z_std = jit(state.apply_fn)(params, f, conditionals, rngs=rngs)
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = norm.logpdf(f, f_hat, 1.0).mean()
            losses[i] = -logp + kl_div.mean()
        if log_results:
            b = [np.array(v) for v in batch]
            p = np.array(f_hat)
            results += [(b, p)]
    loss = losses.mean()
    print(f"validation loss: {loss:.3f}")
    wandb.log({"validation_loss": loss})
    return results


def instantiate(d: Union[dict, DictConfig]):
    """Convenience function to instantiate an object from a config."""
    if isinstance(d, DictConfig):
        d = OmegaConf.to_container(d, resolve=True)
    if "cls" in d:
        cls, kwargs = d["cls"], d.get("kwargs", {})
        for k in kwargs:
            if k == "act_fn":
                kwargs[k] = getattr(nn, kwargs[k])
            elif isinstance(kwargs[k], dict):
                kwargs[k] = instantiate(kwargs[k])
        return globals()[cls](**kwargs)
    elif "func" in d:
        return eval(d["func"])
    return d


if __name__ == "__main__":
    main()

# TODO(jhonathan): delete if not adding new sampling policies to VAE

# def generate_vae_locations(data: DictConfig, rng_train: jax.Array):
#     s_map, sample_grid, s_bounds = process_map(data)
#     if data.sampling_policy == "centroids":
#         return s_map
#     elif data.sampling_policy == "in_map":
#         _, locations_rng = random.split(rng_train)
#         return random.choice(locations_rng, sample_grid, shape=(data.num_ctx.max))
#     elif data.sampling_policy == "grid":
#         ctx_sqrt = int(jnp.sqrt(data.num_ctx.max))
#         return build_grid(
#             [
#                 {"start": s_bounds[0, 0], "stop": s_bounds[0, 1], "num": ctx_sqrt},
#                 {"start": s_bounds[1, 0], "stop": s_bounds[1, 1], "num": ctx_sqrt},
#             ]
#         ).reshape(-1, 2)
#     else:
#         raise ValueError(f"Invalid data sampling policy: {data.sampling_policy}")
