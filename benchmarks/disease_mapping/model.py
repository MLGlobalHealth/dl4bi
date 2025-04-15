"""
Model for malaria prevalence given pointwise observations.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from sps.kernels import _prepare_dims, exponential, rbf

from benchmarks.disease_mapping.utils import haversine_distance, make_pairwise

# TODO @pgrynfelder:
# perhaps use this kernel? https://github.com/malaria-atlas-project/st-cov-fun/blob/master/st_cov_fun.py


# NOTE: RBF is not positive definite on a sphere!
def kernel(x, y, /, *, var, ls, **_):
    """
    Geodesic Laplace (also Exponential, Matern 1/2) kernel.
    """
    # return rbf(x, y, var=var, ls=ls)
    return exponential(x, y, var=var, ls=ls)

    x, y = _prepare_dims(x, y)

    d = make_pairwise(haversine_distance)(x, y)
    return var * jnp.exp(-d / ls)


jitter = 1e-4  # note this is in fact N(0, s2=jitter) independent noise


def spatial_effect(s: jax.Array, *, sample_shape: tuple[int, ...] = ()):
    """
    Definition of the spatial effect underlying the observation model.
    """
    ls = numpyro.sample("ls", dist.InverseGamma(3, 3))
    var = numpyro.deterministic("var", 1)

    cov = kernel(s, s, var=var, ls=ls)
    L = cov.shape[0]
    cov = cov + jitter * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(0, cov),
        sample_shape=sample_shape,
    )

    return y


def prevalence(y):
    b0 = numpyro.sample("b0", dist.Normal(0, 10))
    scale = numpyro.sample("scale", dist.HalfNormal(1))
    logit_theta = numpyro.deterministic("logit_theta", b0 + scale * y)
    numpyro.deterministic("theta", jax.nn.sigmoid(logit_theta))
    return logit_theta


def survey_model(
    s: jax.Array, n_pos: jax.Array, n: jax.Array, *, sample_shape: tuple[int, ...] = ()
):
    """
    MBG survey model.

    Setting `sample_shape` to !=() will produce samples with the same kernel parameters.
    """
    y = spatial_effect(s, sample_shape=sample_shape)
    logit_theta = prevalence(y)

    numpyro.sample(
        "n_pos",
        dist.BinomialLogits(total_count=n, logits=logit_theta),
        obs=n_pos,
    )


def render():
    L = 10
    rng = jax.random.key(0)
    s = jax.random.normal(rng, (L, 2))
    n = jax.random.randint(rng, (L,), 0, 100)
    n_pos = jax.random.randint(rng, (L,), 0, n)
    return numpyro.render_model(
        survey_model,
        (s, n_pos, n),
        render_distributions=True,
        render_params=True,
    )
