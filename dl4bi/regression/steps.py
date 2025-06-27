import jax
from jax import jit, random

from ..core.train import TrainState
from .data import RegressionBatch


@jit
def likelihood_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: RegressionBatch,
    **kwargs,
):
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        output = state.apply_fn(
            {"params": params, **state.kwargs},
            **batch,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        return output.nll(batch["y"])

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def likelihood_valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: RegressionBatch,
    **kwargs,
):
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        training=False,
        rngs={"extra": rng},
    )
    return output.metrics(batch["y"])
