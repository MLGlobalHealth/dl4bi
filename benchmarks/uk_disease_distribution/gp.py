#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import optax
from jax import jit, random
from map_utils import get_raw_map_data, process_map
from omegaconf import DictConfig, OmegaConf
from plot_utils import log_posterior_map_predictive_plots

import wandb
from dl4bi.meta_regression.train_utils import (
    Callback,
    cosine_annealing_lr,
    evaluate,
    instantiate,
    save_ckpt,
    train,
)


def cfg_to_run_name(cfg: DictConfig):
    # TODO(jhonathan): update
    # name = cfg.model.cls
    return cfg.data.sampling_policy + "_test"


def build_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    s_map, sample_grid, s_bounds = process_map(data)
    obs_noise, batch_size = data.obs_noise, data.batch_size
    valid_lens_test = jnp.repeat(data.num_ctx.max + s_map.shape[0], batch_size)

    @jit
    def gen_s_random_centroids(rng: jax.Array):
        return jax.random.choice(rng, s_map, shape=(data.num_ctx.max,))

    @jit
    def gen_s_random_grid(rng: jax.Array):
        rng_x, rng_y = random.split(rng)
        s_x = random.uniform(
            rng_x, (data.num_ctx.max, 1), minval=s_bounds[0, 0], maxval=s_bounds[0, 1]
        )
        s_y = random.uniform(
            rng_y, (data.num_ctx.max, 1), minval=s_bounds[1, 0], maxval=s_bounds[1, 1]
        )
        return jnp.concatenate([s_x, s_y], axis=-1)

    @jit
    def gen_s_random_in_map(rng: jax.Array):
        return jax.random.choice(rng, sample_grid, shape=(data.num_ctx.max,))

    ctx_gen_func = {
        "centroids": gen_s_random_centroids,
        "grid": gen_s_random_grid,
        "in_map": gen_s_random_in_map,
    }[data["sampling_policy"]]

    def gen_batch(rng: jax.Array):
        rng_s_random, rng_valid_lens_ctx, rng_gp, rng_eps, rng = random.split(rng, 5)
        s_random = ctx_gen_func(rng_s_random)
        s = jnp.vstack([s_random, s_map])
        f, var, ls, period, *_ = gp.simulate(rng_gp, s, batch_size)
        valid_lens_ctx = random.randint(
            rng_valid_lens_ctx,
            (batch_size,),
            data.num_ctx.min,
            data.num_ctx.max,
        )
        s = jnp.repeat(s[None, ...], batch_size, axis=0)
        f_noisy = f + obs_noise * random.normal(rng_eps, f.shape)
        return s, f_noisy, valid_lens_ctx, s, f, valid_lens_test, var, ls, period

    def dataloader(rng: jax.Array):
        while True:
            rng_batch, rng = random.split(rng)
            yield gen_batch(rng_batch)

    return dataloader


@hydra.main("configs/gp", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=cfg.get("name", run_name),
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    print(OmegaConf.to_yaml(cfg))

    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    map_data = get_raw_map_data(cfg.data.name)
    dataloader = build_gp_dataloader(cfg.data, cfg.kernel)
    lr_schedule = cosine_annealing_lr(
        cfg.train_num_steps,
        cfg.lr_peak,
        cfg.lr_pct_warmup,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.clip_max_norm),
        optax.yogi(lr_schedule),
    )
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        dataloader,
        dataloader,
        cfg.train_num_steps,
        cfg.valid_num_steps,
        cfg.valid_interval,
        callbacks=[
            Callback(log_posterior_map_predictive_plots(map_data), cfg.plot_interval)
        ],
    )
    metrics = evaluate(rng_test, state, dataloader, cfg.valid_num_steps)
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path = f"results/uk_disease_dist/{cfg.data.name}/{cfg.kernel.kwargs.kernel.func}/{cfg.seed}/{run_name}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


if __name__ == "__main__":
    main()
