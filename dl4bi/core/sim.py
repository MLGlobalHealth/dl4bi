from functools import partial

import jax
import jax.numpy as jnp
from jax import jit
from sps.kernels import l2_dist


@partial(jit, static_argnames=("causal_t",))
def dist_spatiotemporal(
    q: jax.Array,  # [Q, D]
    r: jax.Array,  # [R, D]
    causal_t: bool = False,
):
    """Returns a spatiotemporal distance matrix of shape `[Q, R, 2]`.

    This assumes that the spatial dims of `q` and `r` are all  but the last,
    i.e. `{q,r}[:, :-1]`, and the temporal dim is the last dim,
    i.e. `{q,r}[:, -1]`.

    The entries in `[Q, R, 0]` represent the L2 spatial distance.
    The entries in `[Q, R, 1]` represent the signed L1 temporal distance.

    For example, for `q[i] = [x=2, y=3, t=1]` and `r[j] = [x=1, y=0, t=0]`,
    this will return `m[i, j] = [sqrt(10), -1]`.
    """
    d_s = dist_spatial(q[..., :-1], r[..., :-1])  # [Q, R, 1] L2 dist
    d_t = -q[..., [-1]] + r[..., [-1]].T  # [Q, R] L1 temporal dist
    d = jnp.concatenate([d_s, d_t[..., None]], axis=-1)  # [Q, R, 2]
    if causal_t:  # set distances to inf for future timesteps
        return jnp.where((d_t <= 0)[..., None], d, jnp.inf)
    return d


@jit
def dist_spatial(q: jax.Array, r: jax.Array):
    return l2_dist(q, r)
