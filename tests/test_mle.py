import numpy as np
from jax import random
from sps.gp import GP
from sps.kernels import rbf
from sps.priors import Prior
from sps.utils import build_grid

from dl4bi.core import gp_mle_bfgs


def test_find_gp_mle():
    rng = random.key(55)
    var, ls = 2.0, 0.5
    s = build_grid([{"start": -2.0, "stop": 2.0, "num": 64}] * 2)
    gp = GP(rbf, var=Prior("fixed", {"value": var}), ls=Prior("fixed", {"value": ls}))
    f, *_ = gp.simulate(rng, s)
    f = f[0]  # get rid of batch dim
    var_hat, ls_hat = gp_mle_bfgs(s, f, rbf, jitter=1e-4)
    print(var, var_hat)
    print(ls, ls_hat)
    assert np.isclose(var, var_hat)
    assert np.isclose(ls, ls_hat)
