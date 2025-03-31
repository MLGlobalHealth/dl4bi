"""
Model for malaria prevalence given pointwise observations.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit
from sps.kernels import rbf

from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import TrainState

var_dist = dist.Delta(1)
ls_dist = dist.Beta(3, 7)
kernel = jit(lambda x, y, **kwargs: rbf(x, y, kwargs["ls"], kwargs["var"]))
mean = jit(lambda x, **kwargs: 0)
jitter = 1e-5  # note this is in fact N(0, s2=jitter) noise


# TODO @pgrynfelder:
# perhaps use this kernel? https://github.com/malaria-atlas-project/st-cov-fun/blob/master/st_cov_fun.py
# and a non-0 mean?


def model(
    s: jax.Array,
    Np: jax.Array,
    N: jax.Array,
):
    B, L, D = s.shape

    # isn't it better to account for scale later on in the NP framework? or both?
    var = numpyro.sample("var", var_dist)
    ls = numpyro.sample("ls", ls_dist)

    cov = kernel(s, s, var, ls) + jitter * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(0, cov),
    )

    s = numpyro.sample("s", dist.HalfNormal(50))
    b = numpyro.sample("b", dist.Normal(0, 1))

    logit_theta = b + s * y
    numpyro.deterministic("theta", jax.nn.sigmoid(logit_theta))

    numpyro.sample("Np", dist.BinomialLogits(total_count=N, logits=logit_theta), obs=Np)


@jit
def conditional_gp(s_c: jax.Array, y_c: jax.Array, s_t: jax.Array, **kwargs):
    """
    Get a conditional mean, covariance at s_t sample given an observed (context) sample.
    """
    B, L_ctx, D = s_c.shape
    _, L_test, _ = s_t.shape

    mean_c = mean(s_c, **kwargs)
    mean_t = mean(s_t, **kwargs)

    cov_cc = kernel(s_c, s_c, **kwargs) + jitter * jnp.eye(L_ctx)
    cov_ct = kernel(s_c, s_t, **kwargs)
    cov_tc = cov_ct.mT
    cov_tt = kernel(s_t, s_t, **kwargs) + jitter * jnp.eye(L_test)

    L = jax.scipy.linalg.cho_factor(cov_cc)

    m = mean_t + cov_tc @ jax.scipy.linalg.cho_solve(L, y_c - mean_c)
    cov = cov_tt - cov_tc @ jax.scipy.linalg.cho_solve(L, cov_ct)

    return m, cov


def predict_gp(rng, s_c, y_c, s_t, **kwargs):
    mean, cov = conditional_gp(s_c, y_c, s_t, **kwargs)

    return dist.MultivariateNormal(mean, cov).sample(rng)


@jit(static_argnames="state")
def predict_np(state: TrainState, rng, s_c, y_c, s_t):
    rng, rng_extra = jax.random.split(rng)
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        s_c,
        y_c,
        s_t,
        None,
        rngs={"extra": rng_extra},
    )

    if isinstance(output, tuple):
        output, _ = output

    match output:
        case DiagonalMVNOutput(mu, std):
            return mu + std * jax.random.normal(rng, mu.shape)
        case _:
            raise NotImplementedError()
