from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.experimental import enable_x64
from omegaconf import DictConfig
from sps.kernels import *  # needed by `instantiate`
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


def sample_diagonal(
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
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=1).squeeze(1)

    s_ctx = jnp.expand_dims(s_ctx, 0)
    f_ctx = jnp.expand_dims(f_ctx, 0)
    s_test = jnp.expand_dims(s_test, 0)
    # we only need to expand the dimension to B when multiplying by the random normals

    f_mu, f_std = apply(s_ctx, f_ctx, s_test)
    f_mu = jnp.repeat(f_mu, B, axis=0)
    f_std = jnp.repeat(f_std, B, axis=0)
    f_sampled = normals * f_std + f_mu

    return f_sampled, log_densities


def _sample_autoreg(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [B, L_ctx, D]
    f_ctx: jax.Array,  # [B, L_ctx, 1]
    s_test: jax.Array,  # [B, L_test, D]
):
    B, L_test, _ = s_test.shape
    _, L_ctx, _ = s_ctx.shape

    s_ctx = jnp.concat([s_ctx, s_test], axis=1)
    f_ctx = jnp.pad(f_ctx, ((0, 0), (0, L_test), (0, 0)))

    # note that the independent random normals can be pre-sampled
    # it doesn't matter whether sampling is done here, or in the for loop, as all is independent
    normals = random.normal(rng, (L_test, B))
    log_densities = jnp.sum(jax.scipy.stats.norm.logpdf(normals), axis=0)

    valid_lens_ctx = jnp.repeat(L_ctx, B)

    for i in range(L_test):
        s_test_i = s_test[:, i][:, None]  # [B, 1, 1]

        f_mu_i, f_std_i = apply(
            s_ctx, f_ctx, s_test_i, valid_lens_ctx
        )  # [B, 1, 1], [B, 1, 1]

        f_sampled = normals[i][:, None, None] * f_std_i + f_mu_i
        # need to expand normals[i]'s dims to match f_std_i and f_mu_i

        f_ctx = f_ctx.at[:, L_ctx + i, :].set(f_sampled.squeeze(1))
        valid_lens_ctx = valid_lens_ctx + 1

    return f_ctx[:, L_ctx:], log_densities


def invert_permutation(p: jax.Array):
    return jnp.empty_like(p).at[p].set(jnp.arange(p.size))


def furthest_first(
    s_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, 1]
):
    """
    Order such that always the test point furthest away from the context set
    is considered next when sampling autoregressively.

    TODO: This is O(n^2), perhaps can be done faster?
    """

    # cast to float so that setting to jnp.nan works
    s_ctx = s_ctx.squeeze(-1).astype(jnp.float32)
    s_test = s_test.squeeze(-1).astype(jnp.float32)
    L_test = len(s_test)

    distances = jnp.min(
        vmap(vmap(lambda x, y: jnp.abs(x - y), (None, 0)), (0, None))(s_ctx, s_test),
        axis=0,
    )
    order = []

    for _ in range(L_test):
        i = jnp.nanargmax(distances)
        s_i = s_test[i]

        distances = distances.at[i].set(jnp.nan)
        order.append(i)

        distances = jnp.minimum(distances, jnp.abs(s_test - s_i))

    return jnp.array(order, dtype=jnp.int32)


def closest_first(
    s_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, 1]
):
    """
    Order such that always the test point closest to the context set
    is considered next when sampling autoregressively.
    """
    # cast to float so that setting to jnp.nan works
    s_ctx = s_ctx.squeeze(-1).astype(jnp.float32)
    s_test = s_test.squeeze(-1).astype(jnp.float32)
    L_test = len(s_test)

    distances = jnp.min(
        vmap(vmap(lambda x, y: jnp.abs(x - y), (None, 0)), (0, None))(s_ctx, s_test),
        axis=0,
    )
    order = []

    for _ in range(L_test):
        i = jnp.nanargmin(distances)
        x = s_test[i]

        distances = distances.at[i].set(jnp.nan)
        order.append(i)

        distances = jnp.minimum(distances, jnp.abs(s_test - x))

    return jnp.array(order, dtype=jnp.int32)


def sample_autoreg(
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
        return _sample_autoreg(rng, apply, s_ctx, f_ctx, s_test)
    else:
        _, L_test, D = s_test.shape

        rng, rng_perm = random.split(rng)

        # Locations for each path are permuted independently.
        idx = jnp.repeat(jnp.arange(L_test)[None, :], B, axis=0)
        idx = jax.random.permutation(rng_perm, idx, axis=1, independent=True)

        idx_inv = jax.vmap(invert_permutation)(idx)

        # idx needs to match dimension of array in take_along_axis
        idx = jnp.repeat(idx[:, :, None], D, axis=2)
        idx_inv = idx_inv[:, :, None]

        paths, log_densities = _sample_autoreg(
            rng,
            apply,
            s_ctx,
            f_ctx,
            jnp.take_along_axis(s_test, idx, axis=1),
        )

        return jnp.take_along_axis(paths, idx_inv, axis=1), log_densities


# Probabilistic Machine Learning: An Introduction by Kevin P. Murphy
# Chapter 3.2.3
# Equation 3.28
def analytic_gp(
    s_ctx: jax.Array,  # [L_ctx, 1]
    f_ctx: jax.Array,  # [L_ctx, 1]
    s_test: jax.Array,  # [L_test, 1]
    kernel: Callable,
    var: float,
    ls: float,
    ensure_unique: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """
    Kevin P. Murphy, Probabilistic Machine Learning: An Introduction,
    Chapter 3.2.3,
    Equation 3.28

    Assumes 0 mean,
    and positive-definite covariance matrix `cov_cc = kernel(s_ctx, s_ctx, var, ls)`.
    This is true for the kernels we use modulo repeated locations,
    which can be handled by setting `ensure_unique=True`.
    """
    f_ctx = f_ctx.squeeze(-1)

    if ensure_unique:
        s_ctx, idx, idx_inv = jnp.unique(
            s_ctx, axis=-1, return_index=True, return_inverse=True
        )
        f_ctx = f_ctx[idx]

    # 64-bit precision is required for numerical stability.
    with enable_x64():
        s_ctx = s_ctx.astype(jnp.float64)
        f_ctx = f_ctx.astype(jnp.float64)
        s_test = s_test.astype(jnp.float64)

        cov_tc = kernel(s_test, s_ctx, var, ls)
        cov_ct = cov_tc.T
        cov_cc = kernel(s_ctx, s_ctx, var, ls)
        cov_tt = kernel(s_test, s_test, var, ls)

        # Can't just invert cov_cc or even solve without the positive definite
        # assumption as it leads to huge numerical error. Adding jitter to the diagonal doesn't help.
        # Note that jax.scipy.linalg.solve with assume_a="pos" uses Cholesky decomposition.
        mean = cov_tc @ jax.scipy.linalg.solve(cov_cc, f_ctx, assume_a="pos")
        cov = cov_tt - cov_tc @ jax.scipy.linalg.solve(cov_cc, cov_ct, assume_a="pos")

        mean = mean.astype(jnp.float32)
        cov = cov.astype(jnp.float32)

    if ensure_unique:
        mean = mean[..., idx_inv]
        cov = cov[..., idx_inv][:, idx_inv]

    return mean, cov
