from functools import partial, wraps

import jax
import jax.numpy as jnp
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


def haversine_distance(x, y):
    """Haversine distance.

    x, y: two (Long, Lat) pairs in degrees
    """

    # based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    x_lon, x_lat = x
    y_lon, y_lat = y
    x_lon, x_lat, y_lon, y_lat = map(jnp.deg2rad, (x_lon, x_lat, y_lon, y_lat))

    arc_length = 2 * jnp.arcsin(
        jnp.sqrt(
            jnp.sin((x_lat - y_lat) / 2) ** 2
            + jnp.cos(x_lat) * jnp.cos(y_lat) * jnp.sin((x_lon - y_lon) / 2) ** 2
        )
    )

    return jnp.rad2deg(arc_length)


def great_circle_distance(x, y):
    """
    Given two (longitude, latitude) pairs in degrees,
    returns the great circle distance in degrees.
    """
    # based on https://en.wikipedia.org/wiki/Great-circle_distance#Computational_formulae
    x_lon, x_lat = x
    y_lon, y_lat = y
    x_lon, x_lat, y_lon, y_lat = map(jnp.deg2rad, (x_lon, x_lat, y_lon, y_lat))

    d_lon = jnp.abs(x_lon - y_lon)

    sin = jnp.sin
    cos = jnp.cos

    arc_length = jnp.atan2(
        jnp.sqrt(
            (cos(y_lat) * sin(d_lon)) ** 2
            + (cos(x_lat) * sin(y_lat) - sin(x_lat) * cos(y_lat) * cos(d_lon)) ** 2
        ),
        sin(x_lat) * sin(y_lat) + cos(x_lat) * cos(y_lat) * cos(d_lon),
    )

    return jnp.rad2deg(arc_length)


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


@jit
def zstats_to_tstats(mean, std, n=1000):
    rng = jax.random.key(0)
    z = jax.random.normal(rng, (n, *mean.shape))
    z = z * std + mean
    t = jax.nn.sigmoid(z)
    return t.mean(0), t.std(0)


@jit
def tstats_to_zstats(mean, std, n=1000):
    # this is the less useful direction since t is not normal away from 0.5
    rng = jax.random.key(0)
    t = jax.random.normal(rng, (n, *mean.shape))
    t = t * std + mean
    t = t.clip(0, 1)  # clip to [0,1] to avoid NaNs
    z = jax.scipy.special.logit(t).clip(-1e6, 1e6)
    return z.mean(0), z.std(0)
