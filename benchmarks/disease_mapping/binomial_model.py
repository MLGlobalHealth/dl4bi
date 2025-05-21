"""
RBF binomial observation model.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sps.kernels import rbf


def kernel(x, y, /, *, var, ls, **_):
    return rbf(x, y, var, ls)


jitter = 1e-4  # note this is in fact N(0, s2=jitter) independent noise


def spatial_effect(s: jax.Array):
    """
    Definition of the spatial effect underlying the observation model.

    If sample_shape is not (), the kernel parameters are shared across samples.
    """
    ls = numpyro.sample("ls", dist.InverseGamma(3, 3))
    var = numpyro.sample("var", dist.InverseGamma(3, 3))

    cov = kernel(s, s, var=var, ls=ls)
    L = cov.shape[0]
    cov = cov + jitter * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(0, cov),
    )

    # z == y in this case, record for compat
    z = numpyro.deterministic("z", y)

    return z


def model(
    s: jax.Array,  # [L, Ds]
    n: jax.Array,  # [L]
    n_pos: jax.Array | None,  # [L]
    x: jax.Array | None = None,  # [L, Dx]
):
    z = spatial_effect(s)

    numpyro.sample(
        "n_pos",
        dist.BinomialLogits(total_count=n, logits=z),
        obs=n_pos,
    )
