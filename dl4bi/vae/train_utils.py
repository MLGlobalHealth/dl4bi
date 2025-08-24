from functools import partial
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad

from dl4bi.core.model_output import VAEOutput
from dl4bi.core.train import TrainState


def generate_surrogate_decoder(state: TrainState, model: nn.Module):
    """Wraps a VAE model to issue decoder only calls for sampling

    Args:
        state (TrainState): surrogate model's state
        model (nn.Module): surrogate model object

    Returns: the decoding function
    """

    @jax.jit
    def decoder(z, conditionals, **kwargs):
        return model.apply(
            {"params": state.params, **state.kwargs},
            z,
            conditionals,
            **kwargs,
            method="decode",
        )

    return decoder


@partial(jax.jit, static_argnames=["var_idx"])
def deep_rv_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    var_idx: Optional[int] = None,
):
    """DeepRV training step, MSE(f, f_hat).
    Can be normalized by variance to stabilize training, if
    variance is given as a conditional parameter.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        var_idx: the variance conditional index (if exists)

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def deep_rv_loss(params):
        f, conditionals = batch["f"], batch["conditionals"]
        var = conditionals[var_idx] if var_idx is not None else 1.0
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        return (1 / var) * output.mse(f)

    loss, grads = value_and_grad(deep_rv_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=["var_idx", "weight"])
def inducing_deep_rv_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    var_idx: Optional[int] = None,
    weight: float = 10.0,
):
    """Inducing point DeepRV training step, MSE(K_su @ f, K_su @ f_hat).
    Can be normalized by variance to stabilize training, if
    variance is given as a conditional parameter.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        var_idx: the variance conditional index (if exists)

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def deep_rv_loss(params):
        f_bar_u, conditionals = batch["f"], batch["conditionals"]
        K_su = batch["K_su"]
        var = conditionals[var_idx] if var_idx is not None else 1.0
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        residuals = f_bar_u.squeeze() - output.f_hat.squeeze()
        f_bar_u_mse = (residuals**2).mean()
        f_mse = (jnp.einsum("ij, bj-> bi", K_su, residuals)) ** 2
        return (1 / (var * (1 + weight))) * (f_mse.mean() + weight * f_bar_u_mse)

    loss, grads = value_and_grad(deep_rv_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def elbo_train_step(rng: jax.Array, state: TrainState, batch: dict):
    """Standard VAE training step that uses an ELBO loss.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def elbo_loss(params):
        f = batch["f"]
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        kl_div = output.kl_normal_dist()
        nll = output.nll(f)
        return nll + kl_div

    loss, grads = value_and_grad(elbo_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def prior_cvae_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    mse_weight: float = 1 / 1.8,
):
    """The original PriorCVAE paper's train step.
    mse_weight * mse_loss + kl_divergence.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        mse_weight: weight of mse loss. Defaults to the PriorCVAE
            paper's used value.

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def prior_cvae_loss(params):
        f = batch["f"]
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        kl_div = output.kl_normal_dist()
        mse = output.mse(f)
        return mse_weight * mse + kl_div

    loss, grads = value_and_grad(prior_cvae_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def pi_vae_train_step(rng: jax.Array, state: TrainState, batch: dict):
    def loss_fn(params):
        f = batch["f"]
        f_hat_beta, f_hat_beta_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        loss_1 = optax.squared_error(f_hat_beta, f).mean()
        loss_2 = optax.squared_error(f_hat_beta_hat, f).mean()
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        return loss_1 + loss_2 + kl_div.mean()

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def cond_as_feats(x: jax.Array, cond: jax.Array):
    B, L = x.shape[:2]
    if len(x.shape) == 2:
        x = x[..., None]
    return jnp.concat([x, jnp.tile(cond.flatten(), (B, L, 1))], axis=-1)


@jit
def cond_as_locs(x: jax.Array, cond: jax.Array):
    B, L = x.shape[:2]
    # NOTE: reshape x in case x's shape is [B,L,1]
    return jnp.concat([x.reshape(B, L), jnp.tile(cond.flatten(), (B, 1))], axis=-1)


@partial(jax.jit, static_argnames=["train_num_steps", "warmup_frac", "max_f_w"])
def inducing_deep_rv_train_step_curric(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    train_num_steps: int = 100_000,
    warmup_frac: float = 0.5,
    max_f_w: float = 0.5,
):
    """Curriculum: (1-λ)*MSE(f*,f*_hat) + λ*MSE(K_su f*, K_su f*_hat), λ ramps with step."""

    def loss_fn(params):
        f_bar_u, K_su, step = batch["f"], batch["K_su"], batch["step"]
        output: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        residuals = f_bar_u.squeeze() - output.f_hat.squeeze()
        f_bar_u_mse = (residuals**2).mean()
        f_mse = (jnp.einsum("ij, bj-> bi", K_su, residuals)) ** 2
        # λ ramp: 0 → 1 over warmup_frac * train_num_steps
        warmup_steps = warmup_frac * train_num_steps
        lam = jnp.clip((step / warmup_steps) - 1, 0.0, 1.0) * max_f_w
        loss = (1.0 - lam) * f_bar_u_mse + lam * f_mse.mean()
        return loss

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=["r", "noise_std", "consistency_weight"])
def inducing_deep_rv_train_step_noise(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    r: int = 64,  # how many top right-singular vectors to use
    noise_std: float = 0.05,  # std of noise coefficients in that subspace
    consistency_weight: float = 1.0,  # weight on the consistency term
):
    """Noise injection in top-r right-singular subspace (batch['V_r'] expected as [U, r])."""

    def loss_fn(params):
        f_u_bar, K_su = batch["f"], batch["K_su"]
        # NOTE: r largest eigenvectors, decending order
        eigvecs, _ = jax.lax.linalg.eigh(K_su.T @ K_su)
        V_r = eigvecs[:, -r:][:, ::-1]
        out: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        # Base losses
        residuals = f_u_bar.squeeze() - out.f_hat.squeeze()
        f_u_bar_mse = (residuals**2).mean()
        f_res = jnp.einsum("ij,bj->bi", K_su, residuals)
        f_mse = (f_res**2).mean()
        # Noise in high-gain subspace (same for the whole batch or per-sample; here per-batch):
        rng_noise, _ = jax.random.split(rng)
        eta = noise_std * jax.random.normal(rng_noise, (r,))  # [r]
        epsilon = jnp.einsum("ur,r->u", V_r, eta)  # [U]
        # Noisy target in latent → mapped to f
        f_u_bar_noisy = f_u_bar + epsilon  # broadcast to [B, U]
        f_noisy = jnp.einsum("ij,bj->bi", K_su, f_u_bar_noisy)  # [B, S]
        f_pred = jnp.einsum("ij,bj->bi", K_su, out.f_hat.squeeze())  # [B, S]
        # Consistency: prediction should stay close to noisy target in f-space
        consistency = ((f_pred - f_noisy) ** 2).mean()
        loss = f_mse + f_u_bar_mse + consistency_weight * consistency
        return loss

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=["r", "residual_weight"])
def inducing_deep_rv_train_step_latent_plus_residual(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    r: int = 64,
    residual_weight: float = 1.0,
):
    """Loss = MSE(f*, f*_hat) + residual_weight * ||V_r^T (f* - f*_hat)||^2."""

    def loss_fn(params):
        f_u_bar, K_su = batch["f"], batch["K_su"]
        # NOTE: r largest eigenvectors, decending order
        eigvecs, _ = jax.lax.linalg.eigh(K_su.T @ K_su)
        V_r = eigvecs[:, -r:][:, ::-1]
        out: VAEOutput = state.apply_fn(
            {"params": params}, **batch, rngs={"extra": rng}
        )
        residuals = f_u_bar.squeeze() - out.f_hat.squeeze()  # [B, U]
        f_u_bar_mse = (residuals**2).mean()
        res_top = jnp.einsum("ur,bu->br", V_r, residuals)  # [B, r]
        residual_align = (res_top**2).mean()
        loss = f_u_bar_mse + residual_weight * residual_align
        return loss

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
