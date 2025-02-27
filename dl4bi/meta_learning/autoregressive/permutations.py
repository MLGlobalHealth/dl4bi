import jax
import jax.numpy as jnp
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


# jitting led to a slowdown
# @partial(jit, static_argnames=["n", "batch_size"])
def random_permutations(rng: jax.Array, n: int, batch_size: int):
    """
    Returns array of shape [batch_size, n]
    where each row is a uniformly random permutation of [0, 1, ..., n-1].
    """
    idx = jnp.tile(jnp.arange(n), (batch_size, 1))
    idx = jax.random.permutation(rng, idx, axis=1, independent=False)
    return idx


def ltr(
    s_ctx: jax.Array,  # [L_ctx, D]
):
    """
    Orders such that points are considered left-to-right
    (wrt to the first coordinate if the dimension is >1).
    """
    idx = jnp.argsort(s_ctx, axis=0)[0]  # sort by the first coordinate
    return idx


def invert_permutation(p: jax.Array):
    return jnp.empty_like(p).at[p].set(jnp.arange(p.size))
