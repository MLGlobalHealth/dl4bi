from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from sps.kernels import _prepare_dims, l2_dist

from dl4bi.core.utils import make_pairwise


@partial(jit, static_argnames=("causal",))
def delta_time(
    q: jax.Array,  # [Q, 1]
    r: jax.Array,  # [R, 1]
    causal: bool = True,
):
    d = r.T - q
    if causal:
        return jnp.where(d <= 0, d, jnp.inf)
    return d  # [Q, R]


@jit
def great_circle_dist(x: jax.Array, y: jax.Array) -> jax.Array:
    r"""Great circle distance between two [..., 2] arrays.
    Inputs and outputs are in degrees.

    Args:
        x: Input array of size `[..., 2]`.
        y: Input array of size `[..., 2]`.

    Returns:
        Matrix of all pairwise distances.
    """

    def d(x, y):
        # copied from benchmarks.disease_mapping.utils
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

    x, y = _prepare_dims(x, y)
    return make_pairwise(d)(x, y)
