import jax
from jax import jit, random

from ..core.train import TrainState
from .data.utils import MetaLearningBatch


@jit
def likelihood_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: MetaLearningBatch,
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
        mask = None if batch.mask_test is None else batch.mask_test[..., None]
        return output.nll(batch["f_test"], mask)

    nll, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), nll


@jit
def likelihood_valid_step(
    rng: jax.Array,
    state: TrainState,
    batch: MetaLearningBatch,
    **kwargs,
):
    output = state.apply_fn(
        {"params": state.params, **state.kwargs},
        **batch,
        training=False,
        rngs={"extra": rng},
    )
    if isinstance(output, tuple):
        output, _ = output  # latent output not used here
    mask = None if batch.mask_test is None else batch.mask_test[..., None]
    return output.metrics(batch["f_test"], mask)


@jit
def elbo_train_step(
    rng: jax.Array,
    state: TrainState,
    batch: MetaLearningBatch,
    **kwargs,
):
    rng_dropout, rng_extra = random.split(rng)

    def loss_fn(params):
        output, latent_output_ctx = state.apply_fn(
            {"params": params, **state.kwargs},
            **batch,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        # used by NP family only during training for KL-div loss term
        _, latent_output_test = state.apply_fn(
            {"params": params, **state.kwargs},
            batch["s_test"],
            batch["f_test"],
            batch["s_test"],
            batch.mask_test,
            training=True,
            rngs={"dropout": rng_dropout, "extra": rng_extra},
        )
        mask = None if batch.mask_test is None else batch.mask_test[..., None]
        nll = output.nll(batch["f_test"], mask)
        kl_div = latent_output_ctx.forward_kl_div(latent_output_test)
        return nll + kl_div

    elbo, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), elbo
