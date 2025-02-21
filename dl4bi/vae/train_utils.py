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
        f, var, ls, period, z = batch
        f_hat, z_mu, z_std = state.apply_fn(
            {"params": params}, f, jnp.array([var, ls]), rngs={"extra": rng}
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
        f, var, ls, period, z = batch
        f_hat = state.apply_fn({"params": params}, z, jnp.array([var, ls]))
        return optax.squared_error(f_hat.squeeze(), f.squeeze()).mean()

    loss, grads = value_and_grad(mse_loss)(state.params)
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
