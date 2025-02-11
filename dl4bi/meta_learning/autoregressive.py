from typing import Callable, Literal, Optional
import jax.numpy as jnp
import jax
from omegaconf import OmegaConf, DictConfig
from functools import partial
from collections import deque
from sps.kernels import *
from pathlib import Path
from tqdm import tqdm
from jax.experimental import enable_x64

Apply = Callable[
    [jax.Array, jax.Array, jax.Array, Optional[jax.Array]], (jax.Array, jax.Array)
]


def sample_path(
    rng: jax.Array,
    apply: Apply,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
):
    f_mu, f_std = apply(s_ctx, f_ctx, s_test, None)
    f_sampled = jax.random.normal(rng, f_mu.shape) * f_std + f_mu
    return f_sampled


@partial(jax.jit, static_argnames="apply")
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

    rng = jax.random.split(rng, L_test)
    valid_lens_ctx = jnp.repeat(L_ctx, B)

    for i in tqdm(range(L_test)):
        s_test_i = s_test[:, i : i + 1, :]
        f_mu_i, f_std_i = apply(s_ctx, f_ctx, s_test_i, valid_lens_ctx)
        f_sampled = jax.random.normal(rng[i], f_mu_i.shape) * f_std_i + f_mu_i

        f_ctx = f_ctx.at[:, L_ctx + i : L_ctx + i + 1, :].set(f_sampled)
        valid_lens_ctx = valid_lens_ctx + 1

    return f_ctx[:, L_ctx:]


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
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    strategy: Literal[None, "ltr", "random", "binary"] = None,
):
    match strategy:
        case None:
            return _sample_path_autoreg(rng, apply, s_ctx, f_ctx, s_test)
        # Items in the batch are permuted independently.
        case "random":
            B, L, D = s_test.shape
            rng, rng_perm = jax.random.split(rng)

            idx = jnp.repeat(jnp.arange(L)[None, :], B, axis=0)
            idx = jax.random.permutation(rng_perm, idx, axis=1, independent=True)
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

        # In below cases assume s_test is of dim (B, L, 1),
        # and that it is constant along the first dimension.
        # This can be fixed but incurs a performance cost.
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
            # first sort the data
            idx1 = s_test[0, :, 0].argsort()
            idx1_inv = invert_permutation(idx1)

            # then take the binary ordering
            L = s_test.shape[1]
            idx2 = binary_order(L)
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
):
    """
    Kevin P. Murphy, Probabilistic Machine Learning: An Introduction,
    Chapter 3.2.3,
    Equation 3.28

    Assumes 0 mean.
    """
    f_ctx = f_ctx.squeeze(-1)

    with enable_x64():
        print(s_ctx.dtype, f_ctx.dtype, s_test.dtype)
        s_ctx = s_ctx.astype(jnp.float64)
        f_ctx = f_ctx.astype(jnp.float64)
        s_test = s_test.astype(jnp.float64)

        cov_tc = kernel(s_test, s_ctx, var, ls)
        cov_ct = cov_tc.T
        cov_cc = kernel(s_ctx, s_ctx, var, ls)
        cov_tt = kernel(s_test, s_test, var, ls)
        print(cov_tc.dtype, cov_ct.dtype, cov_cc.dtype, cov_tt.dtype)

        # Can't just invert cov_cc or even solve without the positive definite 
        # assumption as it leads to huge numerical error. Adding jitter to the diagonal doesn't help.
        # Note that jax.scipy.linalg.solve with assume_a="pos" uses Cholesky decomposition.
        mean = cov_tc @ jax.scipy.linalg.solve(cov_cc, f_ctx, assume_a="pos")
        cov = cov_tt - cov_tc @ jax.scipy.linalg.solve(cov_cc, cov_ct, assume_a="pos")

    return mean, cov
