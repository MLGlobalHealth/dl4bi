# %%

import jax
import jax.numpy as jnp
import numpy as np
from sps.kernels import rbf

from benchmarks.meta_learning.gp import build_gp_dataloader
from dl4bi.meta_learning.autoregressive.analytic import (
    analytic_gp,
    analytic_observation_gp,
)
from dl4bi.meta_learning.train_utils import load_ckpt

# %%
L_test = 100
var = 1
ls = 0.3
obs_noise = 0.1
s_test = jnp.linspace(-2, 2, L_test)
s_ctx = jnp.array([-0.1, 0.1])
f_ctx = jnp.array([0.1, -0.2])

mu, cov = analytic_observation_gp(s_ctx, f_ctx, s_test, rbf, var, ls, obs_noise)
# mu, cov = analytic_gp(s_ctx, f_ctx, s_test, rbf, var, ls, obs_noise)
# cov = rbf(s_test, s_test, var, ls) + obs_noise**2 * jnp.eye(L_test)


def entropy(cov):
    D = cov.shape[-1]
    return 0.5 * D * (1 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(cov)[1]


entropy(cov)

# %%
path = "results/1d/rbf/42/TNP-KR: Attention noisy test.ckpt"
_, config = load_ckpt(path)
config.data.batch_size = 1
dataloader = build_gp_dataloader(config.data, config.kernel)(jax.random.key(0))


# %%
def get_entropy():
    num_ctx = config.data.num_ctx.max
    (
        s_ctx,
        f_ctx,
        valid_lens_ctx,
        s_test,
        f_test,
        _valid_lens_test,
        [var],
        [ls],
        period,
    ) = next(dataloader)
    obs_noise = config.data.obs_noise

    s_test = s_test.squeeze()[num_ctx:]
    s_ctx, f_ctx = s_ctx.squeeze(), f_ctx.squeeze()

    mu, cov = analytic_observation_gp(s_ctx, f_ctx, s_test, rbf, var, ls, obs_noise)

    yield entropy(cov)


entropies = []
i = 0
for entropy in get_entropy:
    if i >= 100:
        break
    else:
        i += 1
    entropies.append(entropy)

np.mean(entropies)
# %%
