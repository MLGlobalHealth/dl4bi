from typing import Optional

import jax
import jax.numpy as jnp
from jax import jit, random
from jax.scipy.stats import norm

from ..core import TrainState, mask_from_valid_lens, mvn_logpdf


@jit
def train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        f_mu, f_std = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng},
        )
        return -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def train_step_tril_cov(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression with lower triangular covariances, i.e.
    the L in a Cholesky decomposition.

    .. warning::
        This loss function ignores `valid_lens_test`, since jax doesn't provide
        a `where` mask argument in the multvariate normal's logpdf function.
        This means that the loss returned is taken over all test points for
        every item in the batch. This will produce incorrect results if the
        number of valid test points varies by item in the batch.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        f_mu, f_L = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng},
        )
        B = f_test.shape[0]
        f_test_flat, f_mu_flat = f_test.reshape(B, -1), f_mu.reshape(B, -1)
        nlls = -mvn_logpdf(f_test_flat, f_mu_flat, f_L, is_tril=True)
        # average over L_test to be comparable to diagonal train step loss
        return (nlls / valid_lens_test).mean()

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def fast_attention_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    rng_redraw_random_features: Optional[jax.Array] = None,
    **kwargs,
):
    """Training step for meta regression with diagonal covariances.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        rng_redraw_randoM_features: An optional PRNG key used for redrawing
            random features used in fast attention kernel.

    Returns:
        `TrainState` with updated parameters.
    """

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        (f_mu, f_std), updated_state = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rng_redraw_random_features=rng_redraw_random_features,
            rngs={"dropout": rng},
            mutable=["projections"],
        )
        nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        return nll, updated_state

    (nll, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    return state.apply_gradients(grads=grads, kwargs=updated_state), nll


@jit
def npf_elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Training step for meta regression for Neural Process family with latents.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """
    rng_dropout, rng_latent_z = random.split(rng)

    def loss_fn(params):
        s_ctx, f_ctx, valid_lens_ctx, s_test, f_test, valid_lens_test, *_ = batch
        (B, L_test, _) = s_test.shape
        if valid_lens_test is None:
            valid_lens_test = jnp.repeat(L_test, B)
        mask_test = mask_from_valid_lens(L_test, valid_lens_test)
        f_mu, f_std, z_mu_ctx, z_std_ctx = state.apply_fn(
            {"params": params, **state.kwargs},
            s_ctx,
            f_ctx,
            s_test,
            valid_lens_ctx,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "latent_z": rng_latent_z},
        )
        # used by NP family only during training for KL-div loss term
        _, _, z_mu_test, z_std_test = state.apply_fn(
            {"params": params, **state.kwargs},
            s_test,
            f_test,
            s_test,
            valid_lens_test,
            valid_lens_test,
            training=True,
            rngs={"dropout": rng_dropout, "latent_z": rng_latent_z},
        )
        # KL divergence and NLL assume diagonal covariance, i.e. pointwise.
        # Wikipedia's formulas for MVN KL-div: https://tinyurl.com/wiki-kl-div
        # Tensorflow's diagonal MVN KL-div impl (used here): https://tinyurl.com/diag-kl-div
        # KL( z_dist_test (p) || z_dist_ctx (q) ) =
        diff_log_scale = jnp.log(z_std_test) - jnp.log(z_std_ctx)
        kl_div = (
            0.5 * ((z_mu_test - z_mu_ctx) / z_std_ctx) ** 2
            + 0.5 * jnp.expm1(2 * diff_log_scale)
            - diff_log_scale
        ).sum(axis=-1) / valid_lens_test  # [B] <- avg KL per valid test loc
        nll = -norm.logpdf(f_test, f_mu, f_std).mean(where=mask_test)
        return nll + kl_div.mean()

    elbo, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), elbo
