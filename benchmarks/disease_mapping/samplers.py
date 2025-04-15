import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap
from numpyro.handlers import seed, substitute

from benchmarks.disease_mapping.model import jitter, kernel, prevalence
from benchmarks.disease_mapping.utils import rng_vmap
from dl4bi.core.model_output import DiagonalMVNOutput
from dl4bi.core.train import TrainState


@jit
@rng_vmap
def sample_prevalence(rng, y, **params):
    """Sampling of prevalence given `y` and `params`.
    Args:
        rng: jax.random.PRNGKey
        y: spatial effect of shape `[L_test]`
        params: params to be plugged into the prevalence model.

    Returns:
        logit(prevalence) of shape `[L_test]`
    """
    return seed(substitute(prevalence, params), rng)(y)


@jit
@rng_vmap
def sample_gp(
    rng,
    s_c: jax.Array,  # [L_ctx, D]
    y_c: jax.Array,  # [L_ctx]
    s_t: jax.Array,  # [L_test, D]
    **params,  # passes params to gp mean and kernel
):
    """
    Sample posterior of a mean-0 GP given by params and observations y_c at s_c.

    Time complexity: O(L_ctx^3 + L_test^3)
    """
    L_ctx, _ = s_c.shape
    L_test, _ = s_t.shape

    cov_cc = kernel(s_c, s_c, **params) + jitter * jnp.eye(L_ctx)
    cov_ct = kernel(s_c, s_t, **params)
    cov_tc = cov_ct.mT
    cov_tt = kernel(s_t, s_t, **params) + jitter * jnp.eye(L_test)

    # time O(L_ctx^3)
    cov_cc_cholesky = jsp.linalg.cho_factor(cov_cc)

    # time: O(L_ctx^2) for cho_solve + O(L_test * L_ctx) for matmul
    conditional_mean = cov_tc @ jsp.linalg.cho_solve(cov_cc_cholesky, y_c[..., None])
    # time: O(L_ctx^2 * L_test) for cho_solve + O(L_test^2 * L_ctx) for matmul
    conditional_cov = cov_tt - cov_tc @ jsp.linalg.cho_solve(cov_cc_cholesky, cov_ct)
    # time O(L_test^3) for cholesky
    conditional_cov_cholesky = jsp.linalg.cholesky(conditional_cov, lower=True)
    z = jax.random.normal(rng, (L_test, 1))
    y_t = conditional_mean + conditional_cov_cholesky @ z

    # dim of y was expanded so that batched matmul is straightforward
    return y_t.squeeze(-1)


@jit
@rng_vmap
def sample_gp_pointwise(
    rng,
    s_c: jax.Array,  # [L_ctx, D]
    y_c: jax.Array,  # [L_ctx]
    s_t: jax.Array,  # [L_test, D]
    **params,  # passes params to gp mean and kernel
):
    """Sample pointwise posterior of a mean-0 GP given by params and observations y_c at s_c.

    Time complexity: O(L_ctx^3 + L_test * L_ctx^2)
    """
    L_ctx, _ = s_c.shape
    L_test, _ = s_t.shape

    cov_cc = kernel(s_c, s_c, **params) + jitter * jnp.eye(L_ctx)

    # O(L_ctx^3), can't be reused for different lengthscales
    cov_cc_cholesky = jsp.linalg.cho_factor(cov_cc)

    def calculate_single(s_t):
        s_t = s_t[None]  # [1, D]
        cov_ct = kernel(s_c, s_t, **params)
        cov_tc = cov_ct.mT
        cov_tt = kernel(s_t, s_t, **params) + jitter

        # O(L_ctx^2) for cho_solve + O(L_ctx) for matmul
        conditional_mean = cov_tc @ jsp.linalg.cho_solve(
            cov_cc_cholesky, y_c[..., None]
        )  # [1, 1]

        # O(L_ctx^2) for cho_solve + O(L_ctx) for matmul
        conditional_cov = cov_tt - cov_tc @ jsp.linalg.cho_solve(
            cov_cc_cholesky, cov_ct
        )  # [1, 1]

        return conditional_mean.squeeze(), conditional_cov.squeeze()

    mean, var = vmap(calculate_single)(s_t)
    z = jax.random.normal(rng, (L_test))

    return mean + jnp.sqrt(var) * z


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
            s_ctx=s_c,
            f_ctx=y_c[..., None],
            s_test=s_t,
            rngs={"extra": rng_extra},
        )

        if isinstance(output, tuple):
            output = output[0]

        match output:
            case DiagonalMVNOutput(mu, std):
                mu, std = mu.squeeze(-1), std.squeeze(-1)
                return mu + std * jax.random.normal(rng, mu.shape)
            case _:
                raise NotImplementedError()

    return sample_np
