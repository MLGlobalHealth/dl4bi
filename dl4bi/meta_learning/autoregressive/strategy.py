from enum import StrEnum
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import l2_dist_sq


def furthest_first(
    s_ctx: jax.Array,  # [L_ctx, D]
    s_test: jax.Array,  # [L_test, D]
):
    """
    Order such that always the test point furthest away from the context set
    is considered next.

    Uses L2 distance as the criterion.

    TODO: This is O(n^2), perhaps can be done faster?
    """

    # cast to float so that setting to jnp.nan works
    s_ctx = s_ctx.astype(jnp.float32)
    s_test = s_test.astype(jnp.float32)
    L_test = len(s_test)

    # { dist(s, s_ctx)^2 : s in s_test }
    distances_squared = l2_dist_sq(s_test, s_ctx).min(axis=1)
    order = []

    for _ in range(L_test):
        # pick the furthest point
        i = jnp.nanargmax(distances_squared)
        order.append(i)

        # set the distance to this point to nan to ignore it in future iters
        distances_squared = distances_squared.at[i].set(jnp.nan)

        # update distances
        distances_squared = jnp.minimum(
            distances_squared,
            l2_dist_sq(
                s_test[i][None],  # [1, D]
                s_test,
            ).squeeze(0),  # [1, L_ctx] -> [L_ctx]
        )

    return jnp.array(order, dtype=jnp.int32)


def closest_first(
    s_ctx: jax.Array,  # [L_ctx, D]
    s_test: jax.Array,  # [L_test, D]
):
    """
    Order such that always the test point closest to the context set
    is considered next.

    Uses L2 distance as the criterion.

    TODO: This is O(n^2), perhaps can be done faster?
    """

    # cast to float so that setting to jnp.nan works
    s_ctx = s_ctx.astype(jnp.float32)
    s_test = s_test.astype(jnp.float32)
    L_test = len(s_test)

    # { dist(s, s_ctx)^2 : s in s_test }
    distances_squared = l2_dist_sq(s_test, s_ctx).min(axis=1)
    order = []

    for _ in range(L_test):
        # pick the closest point
        i = jnp.nanargmin(distances_squared)
        order.append(i)

        # set the distance to this point to nan to ignore it in future iters
        distances_squared = distances_squared.at[i].set(jnp.nan)

        # update distances
        distances_squared = jnp.minimum(
            distances_squared,
            l2_dist_sq(
                s_test[i][None],  # [1, D]
                s_test,
            ).squeeze(0),  # [1, L_ctx] -> [L_ctx]
        )

    return jnp.array(order, dtype=jnp.int32)


def left_to_right(
    s: jax.Array,  # [L_ctx, D]
):
    """
    Orders such that points are considered left-to-right
    (wrt to the first coordinate if the dimension is >1).
    """
    idx = jnp.argsort(s, axis=0)[:, 0]  # sort by the first coordinate
    return idx


def invert_permutation(p: jax.Array):
    return jnp.empty_like(p).at[p].set(jnp.arange(p.size))


class Strategy(StrEnum):
    ltr = "ltr"
    random = "random"
    furthest = "furthest"
    closest = "closest"

    @partial(jit, static_argnames=["self", "return_inverse"])
    def get_permutation(self, rng, s_ctx, s_test, return_inverse=False):
        match self:
            case "ltr":
                idx = left_to_right(s_test)
            case "random":
                L_test, _ = s_test.shape
                idx = jax.random.permutation(rng, L_test)
            case "furthest":
                idx = furthest_first(s_ctx, s_test)
            case "closest":
                idx = closest_first(s_ctx, s_test)

        if return_inverse:
            return idx, invert_permutation(idx)
        else:
            return idx

    @partial(jit, static_argnames=["self", "return_inverse"])
    def get_batch_permutation(self, rng, s_ctx, s_test, return_inverse=False):
        B, _, _ = s_ctx.shape
        rng = jax.random.split(rng, B)

        return vmap(self.get_permutation, in_axes=(0, 0, 0, None))(
            rng, s_ctx, s_test, return_inverse
        )
