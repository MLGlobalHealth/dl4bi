from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import jit, random
from sps.kernels import rbf


# TODO(danj): verify this formula
@partial(jit, static_argnames=("jitter", "kernel"))
def condition_gp(
    rng: jax.Array,
    s_ctx: jax.Array,
    f_ctx: jax.Array,
    s_test: jax.Array,
    var: jax.Array,
    ls: jax.Array,
    kernel: Callable = rbf,
    jitter=1e-5,
):
    """Conditions a GP on observed points.

    Source: https://num.pyro.ai/en/stable/examples/gp.html
    """
    L_ctx, L_test = s_ctx.shape[0], s_test.shape[0]
    k_tt = kernel(s_test, s_test, var, ls) + jitter * jnp.eye(L_test)
    k_tc = kernel(s_test, s_ctx, var, ls)
    k_cc = kernel(s_ctx, s_ctx, var, ls) + jitter * jnp.eye(L_ctx)
    K_cc_cho = jax.scipy.linalg.cho_factor(k_cc)
    K = k_tt - k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, k_tc.T)
    mu = k_tc @ jax.scipy.linalg.cho_solve(K_cc_cho, f_ctx)
    noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * random.normal(rng, L_test)
    return mu + noise
