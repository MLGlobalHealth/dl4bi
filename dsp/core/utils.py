import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.training import train_state


class TrainState(train_state.TrainState):
    kwargs: FrozenDict = FrozenDict({})


def mask_from_valid_lens(max_len: int, valid_lens: jax.Array):
    return (jnp.arange(max_len) < valid_lens[..., None])[..., None]
