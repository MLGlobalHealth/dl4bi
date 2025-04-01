#!/usr/bin/env python3
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import wandb
from hydra.utils import instantiate
from jax import jit, random
from omegaconf import DictConfig, OmegaConf
from sps.priors import Prior
from sps.utils import build_grid

from dl4bi.core.train import (
    Callback,
    evaluate,
    save_ckpt,
    train,
)
from dl4bi.meta_learning.data.spatiotemporal import SpatiotemporalData
from dl4bi.meta_learning.utils import (
    cfg_to_run_name,
    regression_to_rgb,
    wandb_2d_img_callback,
)


@hydra.main("configs/bouncing_ball", config_name="default", version_base=None)
def main(cfg: DictConfig):
    run_name = cfg.get("name", cfg_to_run_name(cfg))
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online" if cfg.wandb else "disabled",
        name=run_name,
        project=cfg.project,
        reinit=True,  # allows reinitialization for multiple runs
    )
    path = Path(f"results/{cfg.project}/{cfg.seed}/{run_name}")
    print(OmegaConf.to_yaml(cfg))
    rng = random.key(cfg.seed)
    rng_train, rng_test = random.split(rng)
    dataloader = build_dataloader(cfg.data)
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    clbk = partial(
        wandb_2d_img_callback,
        remap_colors=regression_to_rgb,
        filename_prefix="bouncing_ball",
    )
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
        callbacks=[Callback(clbk, cfg.plot_interval)],
        callback_dataloader=dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


def build_dataloader(data: DictConfig):
    r, sz = data.ball_radius, data.frame_size
    velocity_prior = instantiate(data.velocity_prior)
    position_prior = Prior("uniform", {"minval": r, "maxval": sz - r})
    t = jnp.arange(data.num_frames)
    s = build_grid(data.s)
    s = jnp.repeat(s[None, ...], data.num_frames, axis=0)

    def dataloader(rng: jax.Array):
        while True:
            rng_pos, rng_vel, rng = random.split(rng, 3)
            pos = position_prior.sample(rng_pos, (2,))
            vel = velocity_prior.sample(rng_vel, (2,))
            state = jnp.concat([pos, vel])
            frames = simulate_bouncing_ball(state, r, sz, data.num_frames)
            frames = 2 * (frames - 0.5)  # [0, 1] -> [-1, 1]
            d = SpatiotemporalData(x=None, s=s, t=t, f=frames[..., None])
            for _ in range(data.num_frames // data.batch_size):
                rng_b, rng = random.split(rng)
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


@partial(jit, static_argnames=("ball_radius", "frame_size", "num_frames"))
def simulate_bouncing_ball(
    state: jax.Array,
    ball_radius: int,
    frame_size: int,
    num_frames: int,
):
    r, sz = ball_radius, frame_size

    def step(state: jax.Array, _):
        pos, vel = state[:2], state[2:]
        candidate = pos + vel
        bounce = (candidate - r < 0) | (candidate + r >= sz)
        new_vel = jnp.where(bounce, -vel, vel)
        new_pos = pos + new_vel
        y_coords, x_coords = jnp.indices((sz, sz))
        mask = ((x_coords - new_pos[0]) ** 2 + (y_coords - new_pos[1]) ** 2) <= (r**2)
        frame = mask.astype(jnp.float32)
        new_state = jnp.concat([new_pos, new_vel])
        return new_state, frame

    _, frames = jax.lax.scan(step, state, length=num_frames)
    return frames


if __name__ == "__main__":
    main()
