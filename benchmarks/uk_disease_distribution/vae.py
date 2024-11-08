#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import Generator, Optional

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from graph_model_utils import bym, car, generate_adjacency_matrix, icar, instantiate
from jax import jit, random
from jax.scipy.stats import norm
from map_utils import get_raw_map_data, process_map
from omegaconf import DictConfig, OmegaConf
from plot_utils import log_vae_map_plots
from tqdm import tqdm
from vae_train_utils import elbo_train_step, mse_train_step

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    save_ckpt,
)
from dl4bi.vae import DeepChol, train_utils


@hydra.main("configs", config_name="default_vae", version_base=None)
def main(cfg: DictConfig):
    is_gp = cfg.is_gp
    run_name = cfg.get(
        "name",
        f"VAE_{'GP' if is_gp else cfg.graph_model.name}_{cfg.model.cls}_{cfg.data.sampling_policy}",
    )
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
    model = instantiate(cfg.model)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    if is_gp:
        train_loader, test_loader, cond_names = gp_dataloaders(rng, cfg, s)
    else:
        train_loader, test_loader, cond_names = graph_dataloaders(rng, cfg, map_data)
    state = train(
        rng_train,
        model,
        optimizer,
        isinstance(model, (DeepChol,)),
        train_loader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(
                log_vae_map_plots(map_data, s, cond_names, cfg.model.kwargs.z_dim),
                cfg.plot_interval,
            )
        ],
    )
    results = validate(
        rng_test,
        state,
        isinstance(model, (DeepChol,)),
        test_loader,
        cfg.valid_num_steps,
        log_results=True,
        is_test=True,
    )
    data_gen_name = cfg.kernel.kwargs.kernel.func if is_gp else cfg.graph_model.name
    path = Path(
        f"results/{cfg.project}/{cfg.data.name}/{data_gen_name}/{cfg.seed}/{run_name}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".pkl"), "wb") as save_file:
        pickle.dump(results, save_file)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def gp_dataloaders(
    rng: jax.Array,
    cfg: DictConfig,
    s: jax.Array,
    approx: bool = False,
):
    batch_size = cfg.batch_size
    gp = instantiate(cfg.kernel)
    rng_train, rng_test = random.split(rng)
    conditionals_names = ["var", "ls"]
    if gp.kernel.__name__ == "periodic":
        conditionals_names += ["period"]

    def dataloader(data_rng):
        while True:
            rng_batch, data_rng = random.split(data_rng)
            f, var, ls, period, z = gp.simulate(rng_batch, s, batch_size, approx)
            yield (
                f,
                z,
                ([var, ls] if gp.kernel.__name__ != "periodic" else [var, ls, period]),
            )

    return dataloader(rng_train), dataloader(rng_test), conditionals_names


def graph_dataloaders(rng, cfg, map_data):
    adj_mat = generate_adjacency_matrix(map_data, cfg.graph_construction)
    rng_train, rng_test = random.split(rng)
    dataloader, conditionals_names = {"car": car, "icar": icar, "bym": bym}[
        cfg.graph_model.name
    ](cfg.batch_size, adj_mat, cfg.graph_model)
    return dataloader(rng_train), dataloader(rng_test), conditionals_names


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    is_decoder_only: bool,
    loader: Generator,
    train_num_steps: int = 100000,
    valid_num_steps: int = 25000,
    valid_interval: int = 25000,
    log_every_n: int = 100,
    callbacks: list[Callback] = [],
    state: Optional[TrainState] = None,
):
    rng_params, rng_extra, rng_train = random.split(rng, 3)
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
                state,
                is_decoder_only,
                loader,
                valid_num_steps,
            )
        for cbk in callbacks:
            if (i + 1) % cbk.interval == 0:
                cbk.fn(i, rng_step, state, loader, model)
    return state


def validate(
    rng: jax.Array,
    state: train_utils.TrainState,
    is_decoder_only: bool,
    loader: Generator,
    valid_num_steps: int = 5000,
    log_results: bool = False,
    is_test: bool = False,
):
    _, rng_extra = random.split(rng)
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
    print(f"{'test' if is_test else 'validation'} loss: {loss:.3f}")
    wandb.log({f"{'test' if is_test else 'validation'}": loss})
    return results


if __name__ == "__main__":
    main()
