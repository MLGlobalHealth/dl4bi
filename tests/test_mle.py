import jax.numpy as jnp
import pytest
from jax import random
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from sps.utils import build_grid

from dl4bi.core import gp_mle_bfgs, gp_mle_sgd


@pytest.mark.skip(reason="Difficult to get exact answer from so few data points.")
def test_gp_mle_bfgs():
    rng = random.key(55)
    rng_gp, rng_eps = random.split(rng)
    var, ls, eps = 2.0, 0.5, 0.05
    s = build_grid([{"start": -2.0, "stop": 2.0, "num": 64}] * 2)
    gp = GP(rbf, var=Prior("fixed", {"value": var}), ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng, s)
    f = f[0]  # get rid of batch dim
    f += eps * random.normal(rng_eps, f.shape)
    var_hat, ls_hat, eps_hat = gp_mle_bfgs(s, f, rbf)
    print(var_hat, ls_hat, eps_hat)
    assert jnp.isclose(var, var_hat)
    assert jnp.isclose(ls, ls_hat)


@pytest.mark.skip(reason="Difficult to get exact answer from so few data points.")
def test_gp_mle_sgd():
    rng = random.key(55)
    rng_gp, rng_eps = random.split(rng)
    var, ls, eps = 2.0, 0.5, 0.05
    s = build_grid([{"start": -2.0, "stop": 2.0, "num": 64}] * 2)
    gp = GP(rbf, var=Prior("fixed", {"value": var}), ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng, s)
    f = f[0]  # get rid of batch dim
    f += eps * random.normal(rng_eps, f.shape)
    var_hat, ls_hat, eps_hat = gp_mle_sgd(s, f, rbf)
    print(var_hat, ls_hat, eps_hat)
    assert jnp.isclose(var, var_hat)
    assert jnp.isclose(ls, ls_hat)
    assert jnp.isclose(eps, eps_hat)
