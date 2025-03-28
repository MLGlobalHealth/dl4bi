#!/usr/bin/env python3
from functools import partial
import hydra
import wandb
from time import time
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from hydra.utils import instantiate
from benchmarks.meta_learning.sir import remap_colors
from dl4bi.core.train import (
    Callback,
    evaluate,
    load_ckpt,
    save_ckpt,
    train,
)
import jax
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig
from dl4bi.meta_learning.utils import cfg_to_run_name, wandb_2d_img_callback


def build_dataloader(data: DictConfig):
    # TODO(danj): select frames from each simulation
    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield simulate(
                rng_i,
                data.ball_radius,
                data.num_frames,
                data.frame_size,
                data.velocity_range,
                data.batch_size,
            )

    return dataloader, dataloader, partial(dataloader, is_callback=True)


# @partial(jit, static_argnums=list(range(1, 6)))
def simulate(
    rng: jax.Array,
    ball_radius: int,
    num_frames: int,
    frame_size: tuple[int, int],
    velocity_range: tuple[float, float],
    batch_size: int,
):
    """Simulates red ball movements.

    Args:
        rng: PRNG for jax.random function calls.
        ball_radius: Radius of red ball.
        num_frames: Number of frames per simulation.
        frame_size: Height x width of each frame.
        velocity_range: The min and max values for `vx` and `vy`.
        batch_size: Number of simulations to include per call.

    Returns:
        An array of shape [batch, num_frames, height, width].
    """
    B = batch_size
    R, (H, W), (min_v, max_v) = ball_radius, frame_size, velocity_range
    rng_x, rng_y, rng_vx, rng_vy = random.split(rng, 4)
    x = random.randint(rng_x, (B,), R, W - R)
    y = random.randint(rng_y, (B,), R, H - R)
    vx = random.uniform(rng_vx, (B,), minval=min_v, maxval=max_v)
    vy = random.uniform(rng_vy, (B,), minval=min_v, maxval=max_v)
    frames = []
    for _ in range(num_frames):
        frame = jnp.zeros((B, H, W), dtype=jnp.uint8)
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                if i**2 + j**2 <= R**2:
                    mask_y = (0 <= y + i) & (y + i < H)
                    mask_x = (0 <= x + j) & (x + j < W)
                    mask = mask_x & mask_y

                    if mask.any():
                        _x = (x + j)[mask]
                        _y = (y + i)[mask]
                        frame = frame.at[mask, _y, _x].set(1.0)

        # update position
        x += vx.astype(jnp.int8)
        y += vy.astype(jnp.int8)

        # check for collisions with walls
        collision_x = (x - R <= 0) | (x + R >= W)
        collision_y = (y - R <= 0) | (y + R >= H)

        vx = vx.at[collision_x].multiply(-1)
        vy = vy.at[collision_y].multiply(-1)

        frames.append(frame)

    return jnp.array(frames).transpose(1, 0, 2, 3)


@jit
def bw_to_rgb(frames):
    shape = frames.shape
    frames = frames[..., None]  # [B, N, H, W, 1]
    zeros = jnp.zeros((*shape, 2))  # [B, N, H, W, 2]
    return jnp.concatenate([frames, zeros], axis=-1)


@hydra.main("configs/red_ball", config_name="default", version_base=None)
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

    train_dataloader, valid_dataloader, clbk_dataloader = build_dataloader(
        cfg.data
    )
    optimizer = instantiate(cfg.optimizer)
    model = instantiate(cfg.model)
    if cfg.evaluate_only:
        state, _ = load_ckpt(path.with_suffix(".ckpt"))
        # run once to compile
        evaluate(rng_test, state, model.valid_step, dataloader, num_steps=1)
        start = time()
        metrics = evaluate(
            rng_test,
            state,
            model.valid_step,
            valid_dataloader,
            cfg.valid_num_steps,
        )
        end = time()
        metrics["time_elapsed_s"] = end - start
        return print(metrics)
    clbk = partial(
        wandb_2d_img_callback,
        filename_prefix="sir",
        remap_colors=remap_colors,
        transform_model_output=lambda x: (x.p, x.std),
    )
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
        callbacks=[Callback(clbk, cfg.plot_interval)],
        callback_dataloader=clbk_dataloader,
    )
    metrics = evaluate(
        rng_test,
        state,
        model.valid_step,
        valid_dataloader,
        cfg.valid_num_steps,
    )
    wandb.log({f"Test {m}": v for m, v in metrics.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt(state, cfg, path.with_suffix(".ckpt"))


if __name__ == '__main__':
    rng = random.PRNGKey(42)
    rng_gp, rng_noise = random.split(rng)
    sim = simulate(
        rng,
        1,
        10,
        (6, 6),
        (1, 1),
        3,
    )
    print(sim)
    # main()
