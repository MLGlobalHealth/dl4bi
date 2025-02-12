from collections import deque
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.experimental import enable_x64
from omegaconf import DictConfig
from sps.kernels import *  # needed by `instantiate`
from sps.utils import build_grid
from tqdm import tqdm

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


def sample_path(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
):
    f_mu, f_std = apply(s_ctx, f_ctx, s_test, None)
    f_sampled = random.normal(rng, f_mu.shape) * f_std + f_mu
    return f_sampled


@jit
def normal_log_density(x: float):
    # log( 1/2pi e^(-x^2 / 2) )
    return -jnp.log(2 * jnp.pi) - x**2 / 2


def _sample_path_autoreg(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
):
    B, L_test, D = s_test.shape
    _, L_ctx, _ = s_ctx.shape

    s_ctx = jnp.concat([s_ctx, s_test], axis=1)
    f_ctx = jnp.pad(f_ctx, ((0, 0), (0, L_test), (0, 0)))

    # note that the independent random normals can be pre-sampled
    # it doesn't matter whether sampling is done here, or in the for loop, as all is independent
    normals = random.normal(rng, (L_test, B))
    log_densities = jnp.sum(vmap(normal_log_density)(normals), axis=0)

    valid_lens_ctx = jnp.repeat(L_ctx, B)

    for i in tqdm(range(L_test)):
        s_test_i = s_test[:, i][:, None]  # [B, 1, 1]
        f_mu_i, f_std_i = apply(s_ctx, f_ctx, s_test_i, valid_lens_ctx)
        # [B, 1, 1], [B, 1, 1]

        f_sampled = normals[i][:, None, None] * f_std_i + f_mu_i
        # need to expand normals[i]'s dims to match f_std_i and f_mu_i

        f_ctx = f_ctx.at[:, L_ctx + i, :].set(f_sampled.squeeze(1))
        valid_lens_ctx = valid_lens_ctx + 1

    return f_ctx[:, L_ctx:], log_densities


def invert_permutation(p: jax.Array):
    return jnp.empty_like(p).at[p].set(jnp.arange(p.size))


def binary_order(n: int):
    # produces a permutation like 0, n, n/2, n/4, 3n/4, etc.

    if n <= 2:
        return jnp.arange(n)

    # the ordering to output
    ord = [0, n - 1]
    # stores intervals of the form [l,r] of numbers that are not yet in the result
    seq = deque([(1, n - 2)])

    while len(seq) > 0:
        l, r = seq.popleft()
        m = (l + r) // 2
        ord.append(m)
        if l <= m - 1:
            seq.append((l, m - 1))
        if m + 1 <= r:
            seq.append((m + 1, r))

    return jnp.array(ord)


def sample_path_autoreg(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,  # [L_ctx, D]
    f_ctx: jax.Array,  # [L_ctx, D]
    s_test: jax.Array,  # [L_test, D]
    B: int,  # how many paths to sample
    strategy: Literal[None, "ltr", "random", "binary"] = None,
):
    s_ctx = jnp.repeat(s_ctx[None], B, axis=0)
    f_ctx = jnp.repeat(f_ctx[None], B, axis=0)
    s_test = jnp.repeat(s_test[None], B, axis=0)

    match strategy:
        case None:
            return _sample_path_autoreg(rng, apply, s_ctx, f_ctx, s_test)
        case "random":
            _, L_test, D = s_test.shape
            rng, rng_perm = random.split(rng)

            # Locations for each path are permuted independently.
            idx = jnp.repeat(jnp.arange(L_test)[None, :], B, axis=0)
            idx = random.permutation(rng_perm, idx, axis=1, independent=True)
            idx_inv = jax.vmap(invert_permutation)(idx)

            idx = jnp.repeat(idx[:, :, None], D, axis=2)
            idx_inv = jnp.repeat(idx_inv[:, :, None], D, axis=2)

            sample = _sample_path_autoreg(
                rng,
                apply,
                s_ctx,
                f_ctx,
                jnp.take_along_axis(s_test, idx, axis=1),
            )
            return jnp.take_along_axis(sample, idx_inv, axis=1)

        case "ltr":
            idx = s_test[0, :, 0].argsort()
            idx_inv = invert_permutation(idx)

            return _sample_path_autoreg(
                rng,
                apply,
                s_ctx,
                f_ctx,
                s_test[:, idx, :],
            )[:, idx_inv, :]
        case "binary":
            _, L_test, _ = s_test.shape

            # first sort the data
            idx1 = s_test[0, :, 0].argsort()
            idx1_inv = invert_permutation(idx1)

            # then take the binary ordering
            idx2 = binary_order(L_test)
            idx2_inv = invert_permutation(idx2)

            return _sample_path_autoreg(
                rng,
                apply,
                s_ctx,
                f_ctx,
                s_test[:, idx1, :][:, idx2, :],
            )[:, idx2_inv, :][:, idx1_inv, :]


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
):
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
