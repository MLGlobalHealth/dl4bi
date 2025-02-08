#!/usr/bin/env python3
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, random
from omegaconf import DictConfig


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

    return dataloader


@partial(jit, static_argnums=list(range(1, 6)))
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
                    if 0 <= y + i < H and 0 <= x + j < W:
                        frame[y + i, x + j] = 1.0
        # update position
        x += vx
        y += vy

        # check for collisions with walls
        if x - R <= 0 or x + R >= W:
            vx = -vx
        if y - R <= 0 or y + R >= H:
            vy = -vy

        frames.append(frame)

    return jnp.concatenate(frames, axis=1)


@jit
def bw_to_rgb(frames):
    shape = frames.shape
    frames = frames[..., None]  # [B, N, H, W, 1]
    zeros = jnp.zeros((*shape, 2))  # [B, N, H, W, 2]
    return jnp.concatenate([frames, zeros], axis=-1)
