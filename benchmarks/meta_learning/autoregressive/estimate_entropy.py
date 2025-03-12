# %%
# from dataloader import build_gp_dataloader

import jax
import jax.numpy as jnp
from sps.kernels import rbf

from dl4bi.meta_learning.autoregressive.analytic import (
    analytic_gp,
    analytic_observation_gp,
)

L_test = 100
var = 1
ls = 0.3
obs_noise = 0.1
s_test = jnp.linspace(-2, 2, L_test)
s_ctx = jnp.array([-0.5, 0.5])
f_ctx = jnp.array([0.1, -1])

mu, cov = analytic_observation_gp(s_ctx, f_ctx, s_test, rbf, var, ls, obs_noise)
# cov = rbf(s_test, s_test, var, ls) + obs_noise**2 * jnp.eye(L_test)


def entropy(cov):
    D = cov.shape[-1]
    return 0.5 * D * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(cov)[1]


entropy(cov)

# %%
