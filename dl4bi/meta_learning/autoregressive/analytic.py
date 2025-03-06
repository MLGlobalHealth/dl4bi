from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64


def analytic_gp(
    s_ctx: jax.Array,  # [L_ctx, D] or [L_ctx]
    f_ctx: jax.Array,  # [L_ctx, 1] or [L_ctx]
    s_test: jax.Array,  # [L_test, D] or [L_test]
    kernel: Callable,
    var: float,
    ls: float,
    obs_noise: float,  # observation noise, note this is \sigma, not \sigma^2
) -> tuple[jax.Array, jax.Array]:
    """
    Analytic solution for conditioning a GP based on observations.
    This gives a solution to the **underlying** process, not the observation process.

    Kevin P. Murphy, Probabilistic Machine Learning: An Introduction,
    Equations (17.32 - 17.36)

    Assumes 0 mean,
    and requires the covariance matrix `cov_cc = kernel(s_ctx, s_ctx, var, ls) + obs_noise**2 * I`
    to be strictly positive definite.
    """
    if f_ctx.ndim > 1:
        f_ctx = f_ctx.squeeze(-1)

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

    return mean, cov


def analytic_observation_gp(s_ctx, f_ctx, s_test, kernel, var, ls, obs_noise):
    """
    Analytic solution for the observation process of a GP based on observations.
    This gives a solution to the **observation** process, not the underlying process.
    """
    L_test = s_test.shape[0]
    mean, cov = analytic_gp(s_ctx, f_ctx, s_test, kernel, var, ls, obs_noise)
    return mean, cov + obs_noise**2 * jnp.eye(L_test)
