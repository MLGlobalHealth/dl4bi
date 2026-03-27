#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from flax.core.frozen_dict import FrozenDict
from hydra.utils import instantiate
from jax import lax, random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.tabular import TabularData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/synthetic_air_quality", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,
    )
    print(OmegaConf.to_yaml(cfg))
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_data, rng_train, rng_test = random.split(rng, 3)
    train_dataloader, valid_dataloader, test_dataloader = build_dataloaders(
        rng_data, **cfg.data
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        train_dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        valid_dataloader,
        return_state="best",
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        test_dataloader,
        cfg.test_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloaders(
    rng: jax.Array,
    batch_size: int = 32,
    num_ctx_min: int = 768,  # 48 hours * 16 locations
    num_ctx_max: int = 768,
    num_test: int = 192,  # 12 hours * 16 locations
    total_steps: int = 2400,
    num_locations: int = 16,
    pct_train: float = 0.8,
    pct_valid: float = 0.1,
    pct_test: float = 0.1,
    fixed_effect_scale: float = 1.25,
    plume_scale: float = 1.4,
    plume_ls: float = 0.18,
    weather_ls: float = 0.3,
    humidity_ls: float = 0.24,
    obs_noise: float = 0.1,
):
    if (num_ctx_max + num_test) % num_locations != 0:
        raise ValueError("num_ctx_max + num_test must be divisible by num_locations.")
    if num_ctx_min % num_locations != 0:
        raise ValueError("num_ctx_min must be divisible by num_locations.")
    B = batch_size
    window_steps = (num_ctx_max + num_test) // num_locations
    train, valid, test = load_data(
        rng,
        total_steps=total_steps,
        num_locations=num_locations,
        pct_train=pct_train,
        pct_valid=pct_valid,
        pct_test=pct_test,
        fixed_effect_scale=fixed_effect_scale,
        plume_scale=plume_scale,
        plume_ls=plume_ls,
        weather_ls=weather_ls,
        humidity_ls=humidity_ls,
        obs_noise=obs_noise,
    )

    def build_dataloader(x: jax.Array, s: jax.Array, t: jax.Array, f: jax.Array):
        T, M, D_x = x.shape
        _, D_s = s.shape
        if T < window_steps:
            raise ValueError("Split is shorter than the requested context/test window.")
        s_template = jnp.broadcast_to(s[None, None, ...], (B, window_steps, M, D_s))

        def dataloader(rng: jax.Array):
            while True:
                rng_i, rng_b, rng = random.split(rng, 3)
                start = random.choice(
                    rng_i,
                    T - window_steps + 1,
                    shape=(B, 1),
                    replace=True,
                )
                time_idx = start + jnp.arange(window_steps)[None, :]
                x_batch = x[time_idx]  # [B, W, M, D_x]
                f_batch = f[time_idx]  # [B, W, M, 1]
                t_batch = jnp.broadcast_to(
                    t[time_idx][:, :, None, :],
                    (B, window_steps, M, t.shape[-1]),
                )
                feature_groups = FrozenDict(
                    {
                        "x": x_batch.reshape(B, window_steps * M, D_x),
                        "s": s_template.reshape(B, window_steps * M, D_s),
                        "t": t_batch.reshape(B, window_steps * M, t.shape[-1]),
                    }
                )
                yield TabularData(
                    feature_groups,
                    f_batch.reshape(B, window_steps * M, 1),
                ).batch(
                    rng_b,
                    num_ctx_min,
                    num_ctx_max,
                    num_test,
                    test_includes_ctx=False,
                    forecast=True,
                    t_sorted=True,
                )

        return dataloader

    return build_dataloader(*train), build_dataloader(*valid), build_dataloader(*test)


def load_data(
    rng: jax.Array,
    total_steps: int,
    num_locations: int,
    pct_train: float,
    pct_valid: float,
    pct_test: float,
    fixed_effect_scale: float,
    plume_scale: float,
    plume_ls: float,
    weather_ls: float,
    humidity_ls: float,
    obs_noise: float,
):
    x, s, t, f = generate_panel(
        rng,
        total_steps=total_steps,
        num_locations=num_locations,
        fixed_effect_scale=fixed_effect_scale,
        plume_scale=plume_scale,
        plume_ls=plume_ls,
        weather_ls=weather_ls,
        humidity_ls=humidity_ls,
        obs_noise=obs_noise,
    )
    num_test_steps = int(total_steps * pct_test)
    num_valid_steps = int(total_steps * pct_valid)
    num_train_steps = total_steps - num_valid_steps - num_test_steps
    train = (
        x[:num_train_steps],
        s,
        t[:num_train_steps],
        f[:num_train_steps],
    )
    valid = (
        x[num_train_steps : num_train_steps + num_valid_steps],
        s,
        t[num_train_steps : num_train_steps + num_valid_steps],
        f[num_train_steps : num_train_steps + num_valid_steps],
    )
    test = (
        x[-num_test_steps:],
        s,
        t[-num_test_steps:],
        f[-num_test_steps:],
    )
    x_mu = train[0].mean(axis=(0, 1), keepdims=True)
    x_std = train[0].std(axis=(0, 1), keepdims=True) + 1e-6
    f_mu = train[3].mean(axis=(0, 1), keepdims=True)
    f_std = train[3].std(axis=(0, 1), keepdims=True) + 1e-6

    def standardize(split):
        x_split, s_split, t_split, f_split = split
        return (
            (x_split - x_mu) / x_std,
            s_split,
            t_split,
            (f_split - f_mu) / f_std,
        )

    return standardize(train), standardize(valid), standardize(test)


def generate_panel(
    rng: jax.Array,
    total_steps: int,
    num_locations: int,
    fixed_effect_scale: float,
    plume_scale: float,
    plume_ls: float,
    weather_ls: float,
    humidity_ls: float,
    obs_noise: float,
):
    rng_locs, rng_site, rng_dyn, rng_noise = random.split(rng, 4)
    s = random.uniform(rng_locs, (num_locations, 2), minval=0.05, maxval=0.95)
    site_effect = fixed_effect_scale * random.normal(rng_site, (num_locations,))
    (
        source_center,
        weather_center,
        humidity_center,
        wind_u,
        wind_v,
        temp_bg,
        humidity_bg,
        source_strength,
        regional_bg,
    ) = generate_dynamics(rng_dyn, total_steps)
    rel_source = s[None, ...] - source_center[:, None, :]
    rel_weather = s[None, ...] - weather_center[:, None, :]
    rel_humidity = s[None, ...] - humidity_center[:, None, :]
    dist2_source = jnp.sum(rel_source**2, axis=-1)
    dist2_weather = jnp.sum(rel_weather**2, axis=-1)
    dist2_humidity = jnp.sum(rel_humidity**2, axis=-1)
    plume = plume_scale * source_strength[:, None] * jnp.exp(
        -dist2_source / (2 * plume_ls**2)
    )
    wind = jnp.stack([wind_u, wind_v], axis=-1)[:, None, :]
    source_dir = rel_source / (jnp.sqrt(dist2_source[..., None]) + 1e-3)
    transport = plume * (1.0 + 0.25 * jnp.sum(wind * source_dir, axis=-1))
    temp = temp_bg[:, None] + 0.8 * jnp.exp(-dist2_weather / (2 * weather_ls**2))
    humidity = humidity_bg[:, None] + 0.6 * jnp.exp(
        -dist2_humidity / (2 * humidity_ls**2)
    )
    hours = jnp.arange(total_steps, dtype=jnp.float32)
    hour_angle = 2 * jnp.pi * hours / 24.0
    day_angle = 2 * jnp.pi * hours / (24.0 * 7.0)
    x = jnp.stack(
        [
            temp,
            humidity,
            jnp.broadcast_to(wind_u[:, None], transport.shape),
            jnp.broadcast_to(wind_v[:, None], transport.shape),
            jnp.broadcast_to(jnp.sin(hour_angle)[:, None], transport.shape),
            jnp.broadcast_to(jnp.cos(hour_angle)[:, None], transport.shape),
            jnp.broadcast_to(jnp.sin(day_angle)[:, None], transport.shape),
            jnp.broadcast_to(jnp.cos(day_angle)[:, None], transport.shape),
        ],
        axis=-1,
    )
    noise = obs_noise * random.normal(rng_noise, transport.shape)
    f = (
        site_effect[None, :]
        + transport
        + 0.35 * temp
        - 0.45 * humidity
        + regional_bg[:, None]
        + noise
    )[..., None]
    t = (hours / 24.0)[:, None]
    return x.astype(jnp.float32), s.astype(jnp.float32), t.astype(jnp.float32), f.astype(
        jnp.float32
    )


def generate_dynamics(rng: jax.Array, total_steps: int):
    (
        rng_source_center,
        rng_weather_center,
        rng_humidity_center,
        rng_wind_u,
        rng_wind_v,
        rng_temp_bg,
        rng_humidity_bg,
        rng_source_strength,
        rng_regional_bg,
    ) = random.split(rng, 9)
    source_center = 0.5 + 0.35 * jnp.tanh(
        ar1_series(rng_source_center, total_steps, dim=2, rho=0.985, scale=1.25)
    )
    weather_center = 0.5 + 0.3 * jnp.tanh(
        ar1_series(rng_weather_center, total_steps, dim=2, rho=0.98, scale=1.0)
    )
    humidity_center = 0.5 + 0.25 * jnp.tanh(
        ar1_series(rng_humidity_center, total_steps, dim=2, rho=0.975, scale=1.0)
    )
    hours = jnp.arange(total_steps, dtype=jnp.float32)
    hour_angle = 2 * jnp.pi * hours / 24.0
    wind_u = 0.35 * jnp.sin(hour_angle) + ar1_series(
        rng_wind_u, total_steps, rho=0.96, scale=0.2
    )
    wind_v = 0.35 * jnp.cos(hour_angle + 0.4) + ar1_series(
        rng_wind_v, total_steps, rho=0.95, scale=0.2
    )
    temp_bg = 0.4 * jnp.sin(hour_angle - 0.2) + ar1_series(
        rng_temp_bg, total_steps, rho=0.97, scale=0.2
    )
    humidity_bg = 0.4 * jnp.cos(hour_angle + 0.5) + ar1_series(
        rng_humidity_bg, total_steps, rho=0.97, scale=0.15
    )
    source_strength = 1.0 + ar1_series(
        rng_source_strength, total_steps, rho=0.98, scale=0.35
    )
    regional_bg = ar1_series(rng_regional_bg, total_steps, rho=0.995, scale=0.2)
    return (
        source_center,
        weather_center,
        humidity_center,
        wind_u,
        wind_v,
        temp_bg,
        humidity_bg,
        source_strength,
        regional_bg,
    )


def ar1_series(
    rng: jax.Array,
    total_steps: int,
    dim: int = 1,
    rho: float = 0.98,
    scale: float = 1.0,
):
    eps = random.normal(rng, (total_steps, dim), dtype=jnp.float32)
    rho = jnp.asarray(rho, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32)

    def step(carry, noise):
        nxt = rho * carry + jnp.sqrt(1.0 - rho**2) * scale * noise
        return nxt, nxt

    _, xs = lax.scan(step, jnp.zeros((dim,), dtype=jnp.float32), eps)
    return xs[:, 0] if dim == 1 else xs


if __name__ == "__main__":
    main()
