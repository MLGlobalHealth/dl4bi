#!/usr/bin/env python3
from pathlib import Path
from typing import Generator

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random
from jax.scipy.stats import norm
from map_utils import get_raw_map_data, process_map
from numpyro.distributions import Normal
from omegaconf import DictConfig, OmegaConf
from plot_utils import log_vae_map_plots
from tqdm import tqdm
from vae_train_utils import (
    bym,
    car,
    cholesky,
    elbo_train_step,
    generate_adjacency_matrix,
    generate_model_name,
    get_model_kwargs,
    icar,
    instantiate,
    mse_train_step,
)

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    TrainState,
    cosine_annealing_lr,
    save_ckpt,
)
from dl4bi.vae import train_utils


@hydra.main("configs", config_name="default_vae", version_base=None)
def main(cfg: DictConfig):
    is_gp = cfg.is_gp
    decoder_only = cfg.model.kwargs.get("decoder_only", False)
    model_name = generate_model_name(cfg, is_gp, decoder_only)
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=model_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    map_data = get_raw_map_data(cfg.data.name)
    s, _, _ = process_map(cfg.data)
    model = instantiate(cfg.model)
    vae_kwargs = get_model_kwargs(
        s, cfg.data.pre_process, map_data, cfg.data.graph_construction
    )
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
    if is_gp:
        train_loader, test_loader, large_batch_loader, cond_names = gp_dataloaders(
            rng,
            cfg,
            s,
        )
    else:
        train_loader, test_loader, large_batch_loader, cond_names = graph_dataloaders(
            rng,
            cfg,
            map_data,
        )
    state = train(
        rng_train,
        model,
        optimizer,
        decoder_only,
        train_loader,
        vae_kwargs,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(
                log_vae_map_plots(
                    map_data,
                    s,
                    cond_names,
                    cfg.model.kwargs.z_dim,
                    decoder_only,
                    large_batch_loader,
                    **vae_kwargs,
                ),
                cfg.plot_interval,
            )
        ],
    )
    validate(
        rng_test,
        state,
        decoder_only,
        test_loader,
        vae_kwargs,
        cfg.valid_num_steps,
        is_test=True,
    )
    log_information_completion_score(
        rng, test_loader, state, cfg.model.kwargs.z_dim, vae_kwargs
    )
    data_gen_name = cfg.kernel.kwargs.kernel.func if is_gp else cfg.graph_model.name
    path = Path(
        f"results/{cfg.project}/{cfg.data.name}/{data_gen_name}/{cfg.seed}/{model_name}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def gp_dataloaders(
    rng: jax.Array,
    cfg: DictConfig,
    s: jax.Array,
    large_batch_size: int = 2048,
    jitter: float = 1.1e-4,
):
    n_locations = s.shape[0]
    batch_size = cfg.batch_size
    gp = instantiate(cfg.kernel)
    rng_train, rng_test, rng_large_batch = random.split(rng, 3)
    conditionals_names = ["var", "ls"]
    if gp.kernel.__name__ == "periodic":
        conditionals_names += ["period"]
    z_dist = Normal()

    def dataloader(data_rng, bs=batch_size):
        while True:
            rng_var, rng_ls, rng_period, rng_z, data_rng = random.split(data_rng, 5)
            var = gp.var.sample(rng_var)
            ls = gp.ls.sample(rng_ls)
            if gp.kernel.__name__ == "periodic":
                period = gp.period.sample(rng_period)
                K = gp.kernel(s, s, var, ls, period)
            else:
                K = gp.kernel(s, s, var, ls)
            z = z_dist.sample(rng_z, (bs, n_locations))
            f = cholesky(n_locations, K, z, jitter)[..., None]
            yield (
                f,
                z,
                ([var, ls] if gp.kernel.__name__ != "periodic" else [var, ls, period]),
            )

    return (
        dataloader(rng_train),
        dataloader(rng_test),
        dataloader(rng_large_batch, bs=large_batch_size),
        conditionals_names,
    )


def graph_dataloaders(rng, cfg, map_data, large_batch_size=2048):
    adj_mat = generate_adjacency_matrix(map_data, cfg.data.graph_construction)
    rng_train, rng_test, rng_large_batch = random.split(rng, 3)
    dataloader, conditionals_names = {"car": car, "icar": icar, "bym": bym}[
        cfg.graph_model.name
    ](cfg.batch_size, adj_mat, cfg.graph_model)
    return (
        dataloader(rng_train),
        dataloader(rng_test),
        dataloader(rng_large_batch, bs=large_batch_size),
        conditionals_names,
    )


def train(
    rng: jax.Array,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    is_decoder_only: bool,
    loader: Generator,
    vae_kwargs: dict,
    train_num_steps: int = 100000,
    valid_num_steps: int = 25000,
    valid_interval: int = 25000,
    log_every_n: int = 100,
    callbacks: list[Callback] = [],
):
    rng_params, rng_extra, rng_train = random.split(rng, 3)
    f, z, conditionals = next(loader)
    rngs = {"params": rng_params, "extra": rng_extra}

    x = z if is_decoder_only else f
    train_step = mse_train_step if is_decoder_only else elbo_train_step
    kwargs = model.init(rngs, x, conditionals, **vae_kwargs)
    params = kwargs.pop("params")
    param_count = nn.tabulate(model, rngs)(x, conditionals, **vae_kwargs)
    state = TrainState.create(
        apply_fn=model.apply, params=params, kwargs=kwargs, tx=optimizer
    )
    print(param_count)

    losses = np.zeros((train_num_steps,))
    for i in (pbar := tqdm(range(train_num_steps), unit="batch", dynamic_ncols=True)):
        rng_step, rng_train = random.split(rng_train)
        batch = next(loader)
        state, losses[i] = train_step(rng_step, state, batch, **vae_kwargs)
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
                vae_kwargs,
                valid_num_steps,
            )
        for cbk in callbacks:
            if (i + 1) % cbk.interval == 0:
                cbk.fn(i, rng_step, state, loader)
    return state


def validate(
    rng: jax.Array,
    state: train_utils.TrainState,
    is_decoder_only: bool,
    loader: Generator,
    vae_kwargs: dict,
    valid_num_steps: int = 5000,
    is_test: bool = False,
):
    _, rng_extra = random.split(rng)
    losses = np.zeros((valid_num_steps,))
    mse_score = np.zeros((valid_num_steps,))
    for i in (_ := tqdm(range(valid_num_steps), unit="batch", dynamic_ncols=True)):
        batch = next(loader)
        f, z, conditionals = batch
        params = {"params": state.params, **state.kwargs}
        rngs = {"extra": rng_extra}
        if is_decoder_only:
            f_hat, _, _ = jit(state.apply_fn)(
                params, z, conditionals, **vae_kwargs, rngs=rngs
            )
            losses[i] = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
            mse_score[i] = losses[i]
        else:
            # TODO (jhonathan): combine with train step
            # NOTE: ELBO, assumes normal gaussian
            f_hat, z_mu, z_std = jit(state.apply_fn)(
                params, f, conditionals, **vae_kwargs, rngs=rngs
            )
            kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
            logp = norm.logpdf(f, f_hat, 1.0).mean()
            losses[i] = -logp + kl_div.mean()
            mse_score[i] = optax.squared_error(f_hat, f).mean()
    loss = losses.mean()
    print(f"{'Test' if is_test else 'Validation'} Loss: {loss:.3f}")
    wandb.log({f"{'Test' if is_test else 'Validation'} Loss": loss})
    wandb.log({f"{'Test' if is_test else 'Validation'} MSE": mse_score.mean()})


def log_information_completion_score(
    rng,
    loader,
    state,
    z_dim,
    vae_kwargs,
    num_samples=10,
    num_realizations=1000,
    num_observed=150,
):
    unobserved_mse = np.zeros((num_samples,))
    rng_z, rng_obs, rng_f, rng_dr, rng_ext = random.split(rng, 5)
    f, _, conditionals = next(loader)
    f = random.choice(rng_f, f, (num_samples,), replace=False)
    z = jax.random.normal(rng_z, shape=(num_realizations, z_dim, 1))
    f_hat, _, _ = state.apply_fn(
        {"params": state.params, **state.kwargs},
        z,
        conditionals,
        decode_only=True,
        **vae_kwargs,
        rngs={"dropout": rng_dr, "extra": rng_ext},
    )
    obs_idxs = random.choice(rng_obs, f.shape[1], shape=(num_samples, num_observed))
    for i in range(num_samples):
        f_obs = f[i, obs_idxs[i]].squeeze()
        f_hat_obs = f_hat[:, obs_idxs[i]]
        l2_distances = jnp.mean((f_hat_obs - f_obs) ** 2, axis=1)
        closest_idx = jnp.argmin(l2_distances)
        unobserved_idxs = jnp.setdiff1d(jnp.arange(f.shape[1]), obs_idxs[i])
        f_unobserved = f[i, unobserved_idxs].squeeze()
        f_hat_unobserved = f_hat[closest_idx, unobserved_idxs]
        unobserved_mse[i] = jnp.mean((f_unobserved - f_hat_unobserved) ** 2)
    wandb.log(
        {f"Unobserved MSE for {num_observed} context points": unobserved_mse.mean()}
    )


if __name__ == "__main__":
    main()
