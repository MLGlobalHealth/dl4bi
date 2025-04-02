"""
Model for malaria prevalence given pointwise observations.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit
from sps.kernels import rbf

from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import TrainState

# TODO @pgrynfelder:
# perhaps use this kernel? https://github.com/malaria-atlas-project/st-cov-fun/blob/master/st_cov_fun.py
# and a non-0 mean?

kernel = jit(lambda x, y, **kwargs: rbf(x, y, kwargs["var"], kwargs["ls"]))
mean = jit(lambda x, **kwargs: 0)
jitter = 1e-2  # note this is in fact N(0, s2=jitter) noise


def spatial_process(s: jax.Array, sample_shape: tuple[int, ...] = ()):
    """
    Definition of the spacial process underlying the observation model.
    """
    L, D = s.shape

    ls = numpyro.sample("ls", dist.InverseGamma(3, 3))
    var = numpyro.sample("var", dist.HalfNormal(0.05))
    # noise = numpyro.sample("noise", dist.LogNormal(0.0, 10.0))
    noise = numpyro.deterministic("noise", 0.0)

    m = mean(s)
    cov = kernel(s, s, var=var, ls=ls) + (jitter + noise) * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(m, cov),
        sample_shape=sample_shape,
    )

    return y


def model(
    s: jax.Array,
    Np: jax.Array,
    N: jax.Array,
):
    y = spatial_process(s)

    scale = numpyro.deterministic("scale", 1)
    b0 = numpyro.sample("b0", dist.Normal(0, 1))

    logit_theta = b0 + scale * y
    numpyro.deterministic("theta", jax.nn.sigmoid(logit_theta))

    numpyro.sample(
        "Np",
        dist.BinomialLogits(total_count=N, logits=logit_theta),
        obs=Np,
    )


def render():
    L = 10
    rng = jax.random.key(0)
    s = jax.random.normal(rng, (L, 2))
    N = jax.random.randint(rng, (L,), 0, 100)
    Np = jax.random.randint(rng, (L,), 0, N)
    return numpyro.render_model(
        model,
        (s, Np, N),
        render_distributions=True,
        render_params=True,
    )


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


@partial(jit, static_argnames="state")
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
        output = output[0]

    match output:
        case DiagonalMVNOutput(mu, std):
            return mu + std * jax.random.normal(rng, mu.shape)
        case _:
            raise NotImplementedError()
