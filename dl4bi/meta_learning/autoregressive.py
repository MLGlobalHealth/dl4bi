from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.experimental import enable_x64
from omegaconf import DictConfig
from sps.kernels import l2_dist_sq
from sps.utils import build_grid

from dl4bi.meta_learning.train_utils import instantiate

Apply = Callable[
    [jax.Array, jax.Array, jax.Array, Optional[jax.Array]],
    Tuple[jax.Array, jax.Array],
]


# copied from benchmarks/meta_learning/gp.py
def build_gp_dataloader(data: DictConfig, kernel: DictConfig):
    """Generates batches of GP samples."""
    gp = instantiate(kernel)
    B, S = data.batch_size, len(data.s)
    Nc_min, Nc_max = data.num_ctx.min, data.num_ctx.max
    s_g = build_grid(data.s).reshape(-1, S)  # flatten spatial dims
    L = Nc_max + s_g.shape[0]  # L = num test or all points
    obs_noise, B = data.obs_noise, data.batch_size
    valid_lens_test = jnp.repeat(L, B)
    s_min = jnp.array([axis["start"] for axis in data.s])
    s_max = jnp.array([axis["stop"] for axis in data.s])
    batchify = jit(lambda x: jnp.repeat(x[None, ...], B, axis=0))

    def gen_batch(rng: jax.Array):
        rng_s, rng_gp, rng_v, rng_eps = random.split(rng, 4)
        s_r = random.uniform(rng_s, (Nc_max, S), jnp.float32, s_min, s_max)
        s = jnp.vstack([s_r, s_g])
        f, var, ls, period, *_ = gp.simulate(rng_gp, s, B)
        valid_lens_ctx = random.randint(rng_v, (B,), Nc_min, Nc_max)
        s = batchify(s)
        s_ctx = s[:, :Nc_max, :]
        f_ctx = f + obs_noise * random.normal(rng_eps, f.shape)
        f_ctx = f_ctx[:, :Nc_max, :]
        return s_ctx, f_ctx, valid_lens_ctx, s, f, valid_lens_test, var, ls, period

    def dataloader(rng: jax.Array):
        while True:
            rng_i, rng = random.split(rng)
            yield gen_batch(rng_i)

    return dataloader


def diagonal_sample(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [L_ctx, D]
    f_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, D]
    B: int,  # how many paths to sample
):
    L_test, _ = s_test.shape

    # precompute the randomness
    normals = random.normal(rng, (B, L_test, 1))
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=1)

    # expand the dimensions to B
    s_ctx = jnp.repeat(s_ctx[None], B, axis=0)
    f_ctx = jnp.repeat(f_ctx[None], B, axis=0)
    s_test = jnp.repeat(s_test[None], B, axis=0)

    f_mu, f_std = apply(s_ctx, f_ctx, s_test)
    f_sampled = normals * f_std + f_mu

    # Jacobian transform eps -> f_sampled
    log_densities -= jnp.log(f_std).sum(axis=1)

    return f_sampled, log_densities.squeeze(-1)


def invert_permutation(p: jax.Array):
    return jnp.empty_like(p).at[p].set(jnp.arange(p.size))


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


# this just slowed the code down
# @partial(jit, static_argnames=["n", "batch_size"])
def random_permutations(rng: jax.Array, n: int, batch_size: int):
    """
    Returns array of shape [batch_size, n]
    where each row is a permutation of [0, 1, ..., n-1].
    """
    idx = jnp.tile(jnp.arange(n), (batch_size, 1))
    idx = jax.random.permutation(rng, idx, axis=1, independent=False)
    return idx


def _autoregressive_sample(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [B, L_ctx, D]
    f_ctx: jax.Array,  # [B, L_ctx, 1]
    s_test: jax.Array,  # [B, L_test, D]
):
    B, L_test, _ = s_test.shape
    _, L_ctx, _ = s_ctx.shape

    s = jnp.concat([s_ctx, s_test], axis=1)
    f = jnp.pad(f_ctx, ((0, 0), (0, L_test), (0, 0)))

    # Note that the independent random normals can be pre-sampled.
    # It doesn't matter whether sampling is done here, or in the for loop,
    # as in each iteration the N(0, 1) sampling is independent
    normals = random.normal(rng, (B, L_test))
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=1)

    def loop(i: int, carry: tuple[jax.Array, jax.Array]):
        f, log_densities = carry
        s_test_i = s_test[:, i][:, None]  # [B, 1, D]
        eps = normals[:, i][:, None]  # [B, 1]
        valid_lens_ctx = jnp.repeat(L_ctx + i, B)

        f_mu_i, f_std_i = apply(s, f, s_test_i, valid_lens_ctx)
        f_mu_i, f_std_i = f_mu_i.squeeze(1), f_std_i.squeeze(1)
        f_sampled = eps * f_std_i + f_mu_i

        return (
            f.at[:, L_ctx + i].set(f_sampled),
            # Jacobian transform eps -> f_sampled
            # q(f) = q(eps) * |df/deps|^-1 = q(eps) / f_std, hence in log space subtract log(f_std)
            log_densities - jnp.log(f_std_i.squeeze(1)),
        )

    f, log_densities = jax.lax.fori_loop(
        0,
        L_test,
        loop,
        (f, log_densities),
    )

    return f[:, L_ctx:], log_densities


def autoregressive_sample(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [L_ctx, D]
    f_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, D]
    B: int,  # how many paths to sample
    random: bool = False,  # whether to permute s_test randomly
):
    s_ctx = jnp.repeat(s_ctx[None], B, axis=0)
    f_ctx = jnp.repeat(f_ctx[None], B, axis=0)
    s_test = jnp.repeat(s_test[None], B, axis=0)

    if not random:
        return _autoregressive_sample(rng, apply, s_ctx, f_ctx, s_test)
    else:
        _, L_test, D = s_test.shape

        rng, rng_perm = jax.random.split(rng)

        # Locations for each path are permuted independently.
        idx = random_permutations(rng_perm, L_test, B)
        idx_inv = jax.vmap(invert_permutation)(idx)

        # idx needs to match dimension of array in take_along_axis
        idx = jnp.repeat(idx[..., None], D, axis=-1)
        idx_inv = idx_inv[..., None]
        assert idx.shape == s_test.shape

        paths, log_densities = _autoregressive_sample(
            rng,
            apply,
            s_ctx,
            f_ctx,
            jnp.take_along_axis(s_test, idx, axis=1),
        )

        assert idx_inv.shape == paths.shape
        return jnp.take_along_axis(paths, idx_inv, axis=1), log_densities


def analytic_gp(
    s_ctx: jax.Array,  # [L_ctx, D] or [L_ctx]
    f_ctx: jax.Array,  # [L_ctx, 1] or [L_ctx]
    s_test: jax.Array,  # [L_test, D] or [L_test]
    kernel: Callable,
    var: float,
    ls: float,
    obs_noise: float = 0,  # observation noise, note this is \sigma, not \sigma^2
) -> Tuple[jax.Array, jax.Array]:
    """
    Kevin P. Murphy, Probabilistic Machine Learning: An Introduction,
    Equations (17.32 - 17.36)

    Assumes 0 mean,
    and positive-definite covariance matrix `cov_cc = kernel(s_ctx, s_ctx, var, ls)`.
    This is true for the kernels we use modulo repeated locations,
    which can be handled by setting `ensure_unique=True`.
    """
    L_test = s_test.shape[0]

    if f_ctx.ndim > 1:
        f_ctx = f_ctx.squeeze(-1)

    take_from_context = obs_noise < 1e-2
    if take_from_context:
        print(
            "Assuming no noise, will take values from context set if f_ctx and s_test overlap."
        )
        appear_in_context = []
        for i, s in enumerate(s_test):
            if s in s_ctx:
                appear_in_context.append(i)
        appear_in_context = jnp.array(appear_in_context, dtype=jnp.int32)
        mask = jnp.ones(L_test, dtype=bool).at[appear_in_context].set(False)
        s_test = s_test[mask]

    # 64-bit precision is required for numerical stability.
    with enable_x64():
        s_ctx = s_ctx.astype(jnp.float64)
        f_ctx = f_ctx.astype(jnp.float64)
        s_test = s_test.astype(jnp.float64)

        cov_tc = kernel(s_test, s_ctx, var, ls)
        cov_ct = cov_tc.T
        cov_tt = kernel(s_test, s_test, var, ls)

        # Note that for noisy observations we have to add noise for correctness - see Murphy (17.32 - 17.36).
        cov_cc = kernel(s_ctx, s_ctx, var, ls) + obs_noise**2 * jnp.eye(s_ctx.shape[0])

        # Can't just invert cov_cc or even solve without the positive definite
        # assumption as it leads to huge numerical error. Adding jitter to the diagonal doesn't help.
        # Note that jax.scipy.linalg.solve with assume_a="pos" uses Cholesky decomposition.
        mean = cov_tc @ jax.scipy.linalg.solve(cov_cc, f_ctx, assume_a="pos")
        cov = cov_tt - cov_tc @ jax.scipy.linalg.solve(cov_cc, cov_ct, assume_a="pos")

        mean = mean.astype(jnp.float32)
        cov = cov.astype(jnp.float32)

    if take_from_context:
        mean = jnp.zeros(L_test).at[mask].set(mean).at[~mask].set(f_ctx)
        mask2 = jnp.outer(mask, mask)
        cov = jnp.zeros((L_test, L_test)).at[mask2].set(cov.reshape(-1))

    return mean, cov
