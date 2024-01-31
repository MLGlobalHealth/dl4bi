from collections.abc import Callable
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random
import optax
from flax.training import train_state


@dataclass
class DDIM:
    """Denoising Diffusion Implicit Model (DDIM).

    https://arxiv.org/abs/2010.02502
    """

    beta: jax.Array = jnp.linspace(0.001, 0.02, 100)
    pass

    def __post_init__(self):
        alpha = 1 - self.beta
        alpha_bar = jnp.cumprod(alpha, 0)
        alpha_bar = jnp.concatenate((jnp.array([1.0], alpha_bar[:-1])))
