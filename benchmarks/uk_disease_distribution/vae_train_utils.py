import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training import train_state
from jax import jit, value_and_grad
from jax.scipy.stats import norm


class TrainState(train_state.TrainState):
    kwargs: FrozenDict = FrozenDict({})


@jit
def elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """Standard VAE training step that uses an ELBO loss.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def elbo_loss(params):
        f, _, conditionals = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, conditionals, **kwargs, rngs={"extra": rng}
        )
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        logp = norm.logpdf(f, f_hat, 1.0).mean()
        return -logp + kl_div.mean()

    loss, grads = value_and_grad(elbo_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def mse_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """A VAE decoder-only training step that uses an MSE loss.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def mse_loss(params):
        f, z, conditionals = batch
        f_hat = state.apply_fn({"params": params}, z, conditionals, **kwargs)
        return optax.squared_error(f_hat, f.squeeze()).mean()

    loss, grads = value_and_grad(mse_loss)(state.params)
    return state.apply_gradients(grads=grads), loss


@jit
def prior_cvae_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: tuple,
    **kwargs,
):
    """The original PriorCVAE paper's train step.

    Args:
        rng: A PRNG key.
        state: The current training state.
        batch: Batch of data.

    Returns:
        `TrainState` with updated parameters.
    """

    def elbo_loss(params):
        f, _, conditionals = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, conditionals, **kwargs, rngs={"extra": rng}
        )
        kl_div = -jnp.log(z_std) + (z_std**2 + z_mu**2 - 1) / 2
        mse_loss = optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()
        # TODO(Jhonathan): remove hard-coding
        # TODO change sigma -> check streching  on pre-violin plot
        return (1 / (2 * 0.9)) * mse_loss + kl_div.mean()

    loss, grads = value_and_grad(elbo_loss)(state.params)
    return state.apply_gradients(grads=grads), loss
