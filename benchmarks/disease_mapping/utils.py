from functools import partial, wraps

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap


@partial(jit, static_argnames=["batch_size"])
def batch(x: jax.Array, batch_size: int):
    N, *dims = x.shape
    N_rounded = N // batch_size * batch_size
    if N != N_rounded:
        # this is ok being printed at compile time only
        print(
            f"batch_size does not divide the number of samples. Dropping final {N - N_rounded} samples."
        )
    x = x[:N_rounded].reshape(-1, batch_size, *dims)
    return x


@jit
def unbatch(x: jax.Array):
    N, B, *dims = x.shape
    x = x.reshape(-1, *dims)
    return x


def cartesian_product(*xs):
    n = len(xs)
    return jnp.stack(jnp.meshgrid(*xs), axis=-1).reshape(-1, n)


def map_fn(fn, batch_size=None):
    """
    Like jax.lax.map but a decorator that preserves the function signature.
    Uses jax.lax.map internally.
    """

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        return jax.lax.map(
            lambda x: fn(*x[0], **x[1]), (args, kwargs), batch_size=batch_size
        )

    return wrapped_fn


def rng_vmap(fn):
    """vmap which splits the rng key (assumed to be the first argument)"""

    @wraps(fn)
    def wrapped_fn(rng, *args, **kwargs):
        if args:
            B = args[0].shape[0]
        else:
            B = kwargs.values().__iter__().__next__().shape[0]
        # other arg shapes unsupported for now

        rng = jax.random.split(rng, B)
        return vmap(fn)(rng, *args, **kwargs)

    return wrapped_fn


def z_to_theta(mean, std):
    n = 1000
    rng = jax.random.key(0)
    zs = jax.random.normal(rng, (n, *mean.shape)) * std + mean
    thetas = jax.nn.sigmoid(zs)
    return thetas.mean(axis=0), thetas.std(axis=0)


def theta_to_z(mean, std):
    n = 1000
    rng = jax.random.key(0)
    thetas = jax.random.normal(rng, (n, *mean.shape)) * std + mean
    zs = jsp.special.logit(thetas)
    return zs.mean(axis=0), zs.std(axis=0)
