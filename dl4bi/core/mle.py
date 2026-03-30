"""Gaussian process maximum-likelihood estimation helpers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from jax.scipy.optimize import minimize
from jax.typing import ArrayLike


def gp_mle_bfgs(
    s: ArrayLike,
    f: ArrayLike,
    kernel: Callable,
    initial_var: float = 1.0,
    initial_ls: float = 1.0,
    initial_eps: float = 0.05,
):
    """Estimate GP hyperparameters with BFGS."""
    def nll_fn(theta):
        var, ls, eps = theta
        return gp_nll(s, f, kernel, var, ls, eps)

    return minimize(
        nll_fn,
        jnp.array([initial_var, initial_ls, initial_eps]),
        method="BFGS",
        options=dict(gtol=1e-8),
    ).x  # (var, ls)


def gp_nll(
    s: ArrayLike,
    f: ArrayLike,
    kernel: Callable,
    var: float,
    ls: float,
    noise: float,
):
    """Compute the negative log-likelihood of a Gaussian process."""
    N, D = s.size // s.shape[-1], s.shape[-1]
    s = s.reshape(-1, D)
    f = f.reshape(-1)
    K = kernel(s, s, var, ls) + noise * jnp.eye(N)
    L = cholesky(K)
    S1 = solve_triangular(L, f, lower=True)
    S2 = solve_triangular(L.T, S1, lower=False)
    # TODO(danj): ignore constant terms
    return jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * f @ S2 + 0.5 * N * jnp.log(2 * jnp.pi)


def gp_mle_sgd(
    s: ArrayLike,
    f: ArrayLike,
    kernel: Callable,
    initial_var: float = 1.0,
    initial_ls: float = 1.0,
    initial_eps: float = 0.05,
    loss_tol: float = 1e-4,
    param_tol: float = 1e-5,
    optimizer: optax.GradientTransformation = optax.yogi(learning_rate=1e-3),
    verbose: bool = False,
):
    """Estimate GP hyperparameters with gradient-based optimization."""
    @jax.jit
    def nll_fn(theta):
        var, ls, eps = theta
        return gp_nll(s, f, kernel, var, ls, eps)

    theta = jnp.array([initial_var, initial_ls, initial_eps])
    state = optimizer.init(theta)
    nll = loss_delta = param_delta = jnp.float32("inf")
    while loss_delta > loss_tol or param_delta > param_tol:
        nll_prev = nll
        nll, grad = jax.value_and_grad(nll_fn)(theta)
        updates, state = optimizer.update(grad, state, theta)
        theta_prev = theta
        theta = optax.apply_updates(theta, updates)
        loss_delta = jnp.abs(nll - nll_prev)
        param_delta = jnp.max(jnp.abs(theta - theta_prev))
        if verbose:
            print(
                f"NLL: {nll:0.3f}",
                f"loss_delta: {loss_delta:0.3f}",
                f"param_delta: {param_delta:0.3f}",
            )
    return theta  # (var, ls, noise)
