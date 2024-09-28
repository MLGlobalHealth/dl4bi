"""
Based on reference implementation here: https://krasserm.github.io/2018/03/19/gaussian-processes/
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax.numpy.linalg import cholesky, det, inv
from jax.scipy.linalg import solve_triangular
from jax.scipy.optimize import minimize
from jax.typing import ArrayLike


def find_gp_mle(
    s: ArrayLike,
    f: ArrayLike,
    kernel: Callable,
    initial_var: float = 1.0,
    initial_ls: float = 1.0,
    jitter: float = 1e-6,
):
    return minimize(
        _nll_fn_stable(s, f, kernel, jitter),
        jnp.array([initial_var, initial_ls]),
        method="BFGS",
        options=dict(gtol=1e-8),
    ).x  # (var, ls)


def _nll_fn_stable(s: ArrayLike, f: ArrayLike, kernel: Callable, jitter: float):
    N, D = s.size // s.shape[-1], s.shape[-1]
    s = s.reshape(-1, D)
    f = f.reshape(-1)

    def nll(theta):
        var, ls = theta
        K = kernel(s, s, var, ls) + jitter * jnp.eye(N)
        L = cholesky(K)
        S1 = solve_triangular(L, f, lower=True)
        S2 = solve_triangular(L.T, S1, lower=False)
        # TODO(danj): ignore constant terms
        return (
            jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * f @ S2 + 0.5 * N * jnp.log(2 * jnp.pi)
        )

    return nll


def _nll_fn(s: ArrayLike, f: ArrayLike, kernel: Callable, jitter: float):
    N, D = s.size // s.shape[-1], s.shape[-1]
    s = s.reshape(-1, D)
    f = f.reshape(-1)

    def nll(theta):
        var, ls = theta
        K = kernel(s, s, var, ls) + jitter * jnp.eye(N)
        return (
            0.5 * jnp.log(det(K)) + 0.5 * f @ inv(K) @ f + 0.5 * N * jnp.log(2 * jnp.pi)
        )

    return nll
