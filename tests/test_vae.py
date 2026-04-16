import flax.linen as nn
import jax.numpy as jnp
import optax
from jax import random

from dl4bi.core.mlp import MLP
from dl4bi.core.train import TrainState
from dl4bi.vae import RecursiveGMLPDeepRV
from dl4bi.vae.train_utils import (
    recursive_deep_rv_train_step,
    recursive_deep_rv_valid_step,
)


def test_recursive_gmlp_deep_rv_train_and_valid_steps():
    B, L, D_s, C = 3, 7, 2, 2
    rng = random.key(42)
    rng_s, rng_z, rng_f = random.split(rng, 3)
    batch = {
        "s": random.normal(rng_s, (L, D_s)),
        "z": random.normal(rng_z, (B, L)),
        "conditionals": jnp.ones((C,)),
        "f": random.normal(rng_f, (B, L)),
    }
    model = RecursiveGMLPDeepRV(
        num_cycles=5,
        proj_in=MLP([32, 32], nn.gelu),
        proj_out=MLP([16, 16], nn.gelu),
        embed=MLP([16, 16], nn.gelu),
        head=MLP([24, 1], nn.gelu),
    )
    rng_params, rng_extra = random.split(random.key(0))
    kwargs = model.init({"params": rng_params, "extra": rng_extra}, **batch)
    params = kwargs.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(1e-3),
        kwargs=kwargs,
    )
    output = model.apply({"params": state.params, **state.kwargs}, **batch)
    decoded = model.apply(
        {"params": state.params, **state.kwargs},
        batch["z"],
        batch["conditionals"],
        batch["s"],
        method="decode",
    )

    assert output.f_hat.shape == (5, B, L)
    assert decoded.shape == (5, B, L)
    assert jnp.allclose(output.f_hat, decoded)

    next_state, loss = recursive_deep_rv_train_step(random.key(1), state, batch)
    metrics = recursive_deep_rv_valid_step(random.key(2), state, batch)

    assert int(next_state.step) == int(state.step) + 1
    assert jnp.isfinite(loss)
    assert jnp.allclose(metrics["norm MSE"], metrics["MSE iter5"])
    assert set(metrics) == {
        "norm MSE",
        "MSE iter1",
        "MSE iter2",
        "MSE iter3",
        "MSE iter4",
        "MSE iter5",
        "Delta iter2-iter1",
        "Delta iter3-iter2",
        "Delta iter4-iter3",
        "Delta iter5-iter4",
    }
