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
            {"params": params, **state.kwargs}, **batch, rngs={"extra": rng}
        )
        return (1 / var) * output.mse(f)

    loss, grads = value_and_grad(deep_rv_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=["var_idx", "f_u_bar_w"])
def inducing_deep_rv_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    var_idx: Optional[int] = None,
    f_u_bar_w: float = 10.0,
):
    """Inducing point DeepRV training step,
    (w/(1+w)) * MSE(f, f_hat) + (1/(1+w)) * MSE(K_su @ f, K_su @ f_hat).
    Can be normalized by variance to stabilize training, if
    variance is given as a conditional parameter.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.
        var_idx: the variance conditional index (if exists)
        f_u_bar_w: the weight of the non-projected gp realization mse.

    Returns:
        `TrainState` with updated parameters, and the loss
    """

    def deep_rv_loss(params):
        w = f_u_bar_w
        f_bar_u, conditionals = batch["f"], batch["conditionals"]
        K_su = batch["K_su"]
        var = conditionals[var_idx] if var_idx is not None else 1.0
        output: VAEOutput = state.apply_fn(
            {"params": params, **state.kwargs}, **batch, rngs={"extra": rng}
        )
        residuals = f_bar_u.squeeze() - output.f_hat.squeeze()
        f_bar_u_mse = (residuals**2).mean()
        f_mse = (jnp.einsum("ij, bj-> bi", K_su, residuals)) ** 2
        return (1 / (var * (1 + w))) * (f_mse.mean() + w * f_bar_u_mse)

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
            {"params": params, **state.kwargs}, **batch, rngs={"extra": rng}
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
    mse_weight set as the original paper's default.

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
            {"params": params, **state.kwargs}, **batch, rngs={"extra": rng}
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
            {"params": params, **state.kwargs}, **batch, rngs={"extra": rng}
        )
        loss_1 = optax.squared_error(f_hat_beta, f).mean()
        loss_2 = optax.squared_error(f_hat_beta_hat, f).mean()
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        return loss_1 + loss_2 + kl_div.mean()

    loss, grads = value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=["var_idx"])
def flow_matching_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: dict,
    var_idx: Optional[int] = None,
):
    """Conditional flow matching (CFM) train step.

    Minimises ||v_θ(x_t, t, c) - (f - z)||² where x_t = (1-t)*z + t*f
    and t ~ U(0,1) per sample (straight-path / OT-CFM objective).

    Args:
        rng: PRNG key.
        state: Current training state.
        batch: Dict with keys "f", "z", "conditionals", and optional extras
               ("s", "K", ...) forwarded to the vector field network.
        var_idx: Index into conditionals for variance normalisation.

    Returns:
        Updated TrainState and scalar loss.
    """

    def _loss(params):
        f = batch["f"]
        z0 = batch["z"]
        conditionals = batch["conditionals"]
        B = z0.shape[0]

        rng_t, rng_apply = jax.random.split(rng)
        t = jax.random.uniform(rng_t, (B,))
        t_bc = t[:, None]
        x_t = (1.0 - t_bc) * z0 + t_bc * f
        v_target = f - z0

        extra = {k: v for k, v in batch.items() if k not in ("f", "z", "conditionals")}
        output: VAEOutput = state.apply_fn(
            {"params": params, **state.kwargs},
            x_t,
            conditionals,
            t,
            **extra,
            rngs={"extra": rng_apply},
        )
        scale = conditionals[var_idx] if var_idx is not None else 1.0
        return (1.0 / scale) * jnp.mean((output.f_hat.squeeze() - v_target) ** 2)

    loss, grads = value_and_grad(_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def flow_matching_valid_step(rng: jax.Array, state: TrainState, batch: dict):
    """Validation step for flow matching: measures decoded sample MSE.

    Runs the full ODE decode and compares against the true GP sample.
    This is the metric to monitor — vector field MSE (the training loss)
    and sample MSE are not the same thing.

    Args:
        rng: PRNG key.
        state: Current training state.
        batch: Same format as flow_matching_train_step.

    Returns:
        Dict with "norm MSE".
    """
    z0 = batch["z"]
    f = batch["f"]
    extra = {k: v for k, v in batch.items() if k not in ("f", "z", "conditionals")}
    f_hat = state.apply_fn(
        {"params": state.params, **state.kwargs},
        z0,
        batch["conditionals"],
        **extra,
        rngs={"extra": rng},
        method="decode",
    )
    mse = jnp.mean((f_hat.reshape(f.shape) - f) ** 2)
    return {"norm MSE": mse}


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
