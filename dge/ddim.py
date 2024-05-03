import flax.linen as nn
import jax.numpy as jnp
from jax import Array, random


class DDIM(nn.Module):
    """[Denoising Diffusion Implicit Model](https://arxiv.org/abs/2010.02502).

    Args:
        beta: Variance schedule array.

    Returns:
        An instance of the `DDIM` model.
    """

    beta: Array

    def setup(self):
        self.num_timesteps = self.betas.size
        alpha = jnp.hstack([1.0, jnp.cumprod(1.0 - self.beta)])
        self.sqrt_alpha = jnp.sqrt(alpha)
        self.sqrt_one_minus_alpha = jnp.sqrt(1 - alpha)
        self.sqrt_alpha_ratio = jnp.sqrt(alpha[:-1] / alpha[1:])
        self.reverse_variance = jnp.sqrt(
            alpha[:-1] * (1 - alpha[1:]) / alpha[1:]
        ) - jnp.sqrt(1 - alpha[:-1])

    def noise(self, key: Array, x_0: Array, t: int):
        noise = random.normal(key, x_0.shape)
        x_t = self.sqrt_alpha[t] * x_0 + self.sqrt_one_minus_alpha[t] * noise
        return x_t, noise

    def denoise(self, key, x_t: Array, t: int):
        pass

    def __call__(self, x):
        pass
