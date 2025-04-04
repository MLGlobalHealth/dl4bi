"""
Model for malaria prevalence given pointwise observations.
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import jit, vmap
from numpyro.handlers import seed, substitute
from sps.kernels import rbf

from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import TrainState

# TODO @pgrynfelder:
# perhaps use this kernel? https://github.com/malaria-atlas-project/st-cov-fun/blob/master/st_cov_fun.py
# and a non-0 mean?

kernel = jit(lambda x, y, /, **params: rbf(x, y, params["var"], params["ls"]))
mean = jit(lambda x, /, **params: 0)
jitter = 1e-4  # note this is in fact N(0, s2=jitter) noise


def spatial_effect(s: jax.Array, sample_shape: tuple[int, ...] = ()):
    """
    Definition of the spatial effect underlying the observation model.
    """
    L, D = s.shape

    ls = numpyro.sample("ls", dist.InverseGamma(3, 3))
    var = numpyro.sample("var", dist.HalfNormal(1))
    noise = numpyro.deterministic("noise", 0.0)

    m = mean(s)
    cov = kernel(s, s, var=var, ls=ls) + (jitter + noise) * jnp.eye(L)

    y = numpyro.sample(
        "y",
        dist.MultivariateNormal(m, cov),
        sample_shape=sample_shape,
    )

    return y


def prevalence(y):
    b0 = numpyro.sample("b0", dist.Normal(0, 10))

    logit_theta = b0 + y
    numpyro.deterministic("theta", jax.nn.sigmoid(logit_theta))
    return logit_theta


def survey_model(s: jax.Array, np: jax.Array, n: jax.Array):
    y = spatial_effect(s)
    logit_theta = prevalence(y)

    numpyro.sample(
        "np",
        dist.BinomialLogits(total_count=n, logits=logit_theta),
        obs=np,
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


def conditional_gp(
    s_c: jax.Array,  # [L_ctx, D]
    y_c: jax.Array,  # [L_ctx]
    s_t: jax.Array,  # [L_test, D]
    **params,
):
    """
    Get a conditional mean, covariance at s_t sample given an observed (context) sample.
    """
    print(s_c.shape, y_c.shape, s_t.shape)
    for k, v in params.items():
        print(k, v.shape)

    L_ctx, D = s_c.shape
    L_test, _ = s_t.shape

    mean_c = mean(s_c, **params)
    mean_t = mean(s_t, **params)

    cov_cc = kernel(s_c, s_c, **params) + jitter * jnp.eye(L_ctx)
    cov_ct = kernel(s_c, s_t, **params)
    cov_tc = cov_ct.mT
    cov_tt = kernel(s_t, s_t, **params) + jitter * jnp.eye(L_test)

    L = jax.scipy.linalg.cho_factor(cov_cc)

    m = mean_t + cov_tc @ jax.scipy.linalg.cho_solve(L, y_c - mean_c)
    cov = cov_tt - cov_tc @ jax.scipy.linalg.cho_solve(L, cov_ct)

    return m, cov


@jit
def sample_gp(
    rng,
    s_c: jax.Array,  # [B, L_ctx, D]
    y_c: jax.Array,  # [B, L_ctx]
    s_t: jax.Array,  # [B, L_test, D]
    **params,  # passes params to gp mean and kernel, each of dim [B, ...]
):
    """Batched GP conditioning."""
    mean, cov = vmap(conditional_gp)(s_c, y_c, s_t, **params)

    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(rng, mean.shape)

    return mean + jnp.einsum("bij,bj->bi", L, z)


def get_np_sampler(
    state: TrainState,
):
    @jit
    def sample_np(
        rng,
        s_c: jax.Array,  # [B, L, D]
        y_c: jax.Array,  # [B, L]
        s_t: jax.Array,  # [B, L, D]
        **params,  # ignored
    ):
        """Batched Neural Process sampler"""
        rng, rng_extra = jax.random.split(rng)
        output = state.apply_fn(
            {"params": state.params, **state.kwargs},
            s_c,
            y_c[:, None],
            s_t,
            None,
            rngs={"extra": rng_extra},
        )

        if isinstance(output, tuple):
            output = output[0]

        match output:
            case DiagonalMVNOutput(mu, std):
                assert mu.ndim == 2 and std.ndim == 2

                return mu + std * jax.random.normal(rng, mu.shape)
            case _:
                raise NotImplementedError()

    return sample_np


@jit
def sample_prevalence(rng, y, **params):
    """Batched sampling of prevalence given `y` and `params`.
    Args:
        rng: jax.random.PRNGKey
        y: spatial effect
        params: params to be plugged into the prevalence model.

    Returns:
        logit(prevalence)
    """
    rng = jax.random.split(rng, y.shape[0])
    return vmap(lambda rng, y, params: seed(substitute(prevalence, params), rng)(y))(
        rng, y, params
    )
