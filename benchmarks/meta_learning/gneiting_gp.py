#!/usr/bin/env python3
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf

from dl4bi.core.train import (
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning import SGNP
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
from dl4bi.meta_learning.utils import cfg_to_run_name


@hydra.main("configs/gneiting_gp", config_name="default", version_base=None)
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
    path = Path(f"results/{cfg.project}/{cfg.data.name}/{cfg.seed}/{run_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.key(cfg.seed)
    rng_graph, rng_train, rng_test = random.split(rng, 3)
    dataloader = build_dataloader(cfg.data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    if not cfg.data.random_t and isinstance(model, SGNP):
        batch = next(dataloader(rng_graph))
        model = instantiate(cfg.model, graph=model.build_graph(**batch))
    output_fn = model.output_fn
    model = model.copy(output_fn=lambda x: output_fn(x, min_std=0.05))
    state = train(
        rng_train,
        model,
        optimizer,
        model.train_step,
        cfg.train_num_steps,
        dataloader,
        model.valid_step,
        cfg.valid_interval,
        cfg.valid_num_steps,
        dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig):
    if data.num_t > data.num_steps:
        raise ValueError("num_t must be <= num_steps.")
    s_grid = build_grid(data.s)
    spatial_shape = s_grid.shape[:-1]
    num_locations = s_grid.reshape(-1, s_grid.shape[-1]).shape[0]
    t = jnp.linspace(data.t.start, data.t.stop, data.num_steps, dtype=jnp.float32)

    def dataloader(rng: jax.Array):
        while True:
            rng_s, rng_f, rng_b, rng = random.split(rng, 4)
            if data.irregular_layout:
                s_flat = sample_jittered_grid(
                    rng_s,
                    data.s,
                    jitter_frac=data.spatial_jitter_frac,
                )
                spatial_shape_i = (num_locations,)
            else:
                s_flat = s_grid.reshape(-1, s_grid.shape[-1])
                spatial_shape_i = spatial_shape
            s = jnp.repeat(s_flat[None, ...], data.num_steps, axis=0)
            f = sample_gneiting_field(
                rng_f,
                s_flat,
                t,
                spatial_shape_i,
                var=data.var,
                ls_min=data.ls_min,
                ls_max=data.ls_max,
                a_min=data.a_min,
                a_max=data.a_max,
                alpha_min=data.alpha_min,
                alpha_max=data.alpha_max,
                b=data.b,
                nu_min=data.nu_min,
                nu_max=data.nu_max,
                spatial_mean_scale=data.spatial_mean_scale,
                temporal_mean_scale=data.temporal_mean_scale,
                interaction_mean_scale=data.interaction_mean_scale,
                jitter=data.jitter,
                obs_noise=data.obs_noise,
            )
            d = SpatiotemporalData(x=None, s=s, t=t, f=f)
            yield d.batch(
                rng_b,
                data.num_t,
                data.random_t,
                data.num_ctx_per_t.min,
                data.num_ctx_per_t.max,
                data.independent_t_masks,
                data.num_test,
                data.forecast,
                data.batch_size,
            )

    return dataloader


def build_grid(axes: list[dict]):
    coords = [
        jnp.linspace(axis["start"], axis["stop"], axis["num"], dtype=jnp.float32)
        for axis in axes
    ]
    return jnp.stack(jnp.meshgrid(*coords, indexing="ij"), axis=-1)


def sample_jittered_grid(
    rng: jax.Array,
    axes: list[dict],
    jitter_frac: float,
):
    base = build_grid(axes)
    noise = random.uniform(rng, base.shape, minval=-1.0, maxval=1.0)
    steps = []
    for axis in axes:
        num = axis["num"]
        if num <= 1:
            steps.append(0.0)
        else:
            steps.append((axis["stop"] - axis["start"]) / (num - 1))
    step = jnp.asarray(steps, dtype=jnp.float32)
    lo = jnp.asarray([axis["start"] for axis in axes], dtype=jnp.float32)
    hi = jnp.asarray([axis["stop"] for axis in axes], dtype=jnp.float32)
    jitter = noise * (jitter_frac * step)
    return jnp.clip(base + jitter, lo, hi).reshape(-1, len(axes))


def sample_gneiting_field(
    rng: jax.Array,
    s: jax.Array,
    t: jax.Array,
    spatial_shape: tuple[int, ...],
    var: float,
    ls_min: float,
    ls_max: float,
    a_min: float,
    a_max: float,
    alpha_min: float,
    alpha_max: float,
    b: float,
    nu_min: float,
    nu_max: float,
    spatial_mean_scale: float,
    temporal_mean_scale: float,
    interaction_mean_scale: float,
    jitter: float,
    obs_noise: float,
):
    rng_h, rng_z, rng_eps = random.split(rng, 3)
    ls, a, alpha, nu = sample_hyperparams(
        rng_h,
        ls_min,
        ls_max,
        a_min,
        a_max,
        alpha_min,
        alpha_max,
        nu_min,
        nu_max,
    )
    K = gneiting_covariance_matrix(s, t, s, t, var, ls, a, alpha, b, nu)
    K += jitter * jnp.eye(K.shape[0], dtype=jnp.float32)
    z = random.normal(rng_z, (K.shape[0],), dtype=jnp.float32)
    mean = deterministic_mean(
        s,
        t,
        spatial_mean_scale,
        temporal_mean_scale,
        interaction_mean_scale,
    )
    f = mean + jnp.linalg.cholesky(K).astype(jnp.float32) @ z
    if obs_noise > 0.0:
        f += obs_noise * random.normal(rng_eps, f.shape, dtype=jnp.float32)
    return f.reshape(t.shape[0], *spatial_shape, 1)


def sample_hyperparams(
    rng: jax.Array,
    ls_min: float,
    ls_max: float,
    a_min: float,
    a_max: float,
    alpha_min: float,
    alpha_max: float,
    nu_min: float,
    nu_max: float,
):
    rng_ls, rng_a, rng_alpha, rng_nu = random.split(rng, 4)
    ls = sample_interval(rng_ls, ls_min, ls_max)
    a = sample_interval(rng_a, a_min, a_max)
    alpha = sample_interval(rng_alpha, alpha_min, alpha_max)
    nu = sample_interval(rng_nu, nu_min, nu_max)
    return ls, a, alpha, nu


def sample_interval(rng: jax.Array, low: float, high: float):
    low = jnp.asarray(low, dtype=jnp.float32)
    high = jnp.asarray(high, dtype=jnp.float32)
    return jnp.where(
        jnp.isclose(low, high),
        low,
        random.uniform(rng, (), minval=low, maxval=high),
    )


def deterministic_mean(
    s: jax.Array,
    t: jax.Array,
    spatial_mean_scale: float,
    temporal_mean_scale: float,
    interaction_mean_scale: float,
):
    sx, sy = s[:, 0], s[:, 1]
    spatial = (
        0.8 * jnp.sin(1.5 * jnp.pi * sx)
        + 0.5 * jnp.cos(1.0 * jnp.pi * sy)
        + 0.7 * jnp.exp(-((sx - 0.35) ** 2 + (sy + 0.2) ** 2) / (2 * 0.22**2))
        - 0.6 * jnp.exp(-((sx + 0.45) ** 2 + (sy - 0.3) ** 2) / (2 * 0.18**2))
    )
    temporal = jnp.sin(2 * jnp.pi * t + 0.35) + 0.35 * jnp.cos(4 * jnp.pi * t - 0.15)
    interaction = (
        jnp.sin(jnp.pi * sx)[None, :] * jnp.cos(2 * jnp.pi * t[:, None] + 0.2)
        + 0.5 * jnp.cos(jnp.pi * sy)[None, :] * jnp.sin(2 * jnp.pi * t[:, None] - 0.4)
    )
    return (
        spatial_mean_scale * spatial[None, :]
        + temporal_mean_scale * temporal[:, None]
        + interaction_mean_scale * interaction
    ).reshape(-1)


@jit
def gneiting_covariance_matrix(
    s1: jax.Array,
    t1: jax.Array,
    s2: jax.Array,
    t2: jax.Array,
    var: float,
    ls: float,
    a: float,
    alpha: float,
    b: float,
    nu: float,
):
    """Gneiting (2002) non-separable space-time covariance on R^d x R."""
    L1, T1 = s1.shape[0], t1.shape[0]
    L2, T2 = s2.shape[0], t2.shape[0]
    h2 = jnp.sum((s1[:, None, :] - s2[None, :, :]) ** 2, axis=-1)
    u = jnp.abs(t1[:, None] - t2[None, :])
    h2 = h2[None, None, :, :]
    u = u[:, :, None, None]
    g = 1.0 + a * u ** (2 * alpha)
    K = var / (g**nu) * jnp.exp(-h2 / (ls**2 * g**b))
    return K.transpose(0, 2, 1, 3).reshape(T1 * L1, T2 * L2)


if __name__ == "__main__":
    main()
