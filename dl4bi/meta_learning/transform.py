from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax, softplus


@partial(jit, static_argnames=("min_std",))
def diagonal_mvn(f_dist: jax.Array, min_std: float = 0.0):
    f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
    f_std = min_std + (1 - min_std) * softplus(f_std)
    return f_mu, f_std


@partial(jit, static_argnames=("min_std",))
def latent_diagonal_mvn(f_dist: jax.Array, min_std: float = 0.0):
    f_mu, f_std = jnp.split(f_dist, 2, axis=-1)
    f_std = min_std + (1 - min_std) * softplus(f_std)
    return f_mu.mean(axis=1), f_std.mean(axis=1)  # average over n_z latent samples


@jit
def identity(output: jax.Array):
    return output


@jit
def latent_logits(logits: jax.Array):
    return logits.mean(axis=1)  # average over n_z latent samples


@jit
def pointwise_multinomial(f_dist: jax.Array):
    p = softmax(f_dist, axis=-1)
    return p, p * (1 - p)
