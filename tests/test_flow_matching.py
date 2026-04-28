import jax.numpy as jnp
from jax import random

from dl4bi.vae import FlowMatchingDeepRV, FlowMatchingVectorField


def test_flow_matching_init_with_loader_batch():
    """Regression: ``model.init(rngs, **batch)`` must work with the same batch
    shape the training dataloaders yield (``f, z, conditionals, s`` — no ``t``).

    ``dl4bi.core.train.train`` calls ``model.init(rngs, **batch)`` to set up
    params; if ``t`` is required, init crashes before the first training step.
    """
    B, L, d = 4, 16, 2
    rng = random.key(0)
    rng_init, rng_extra, rng_z, rng_s, rng_f = random.split(rng, 5)

    batch = {
        "s": random.uniform(rng_s, (L, d)),
        "z": random.normal(rng_z, (B, L)),
        "conditionals": jnp.array([20.0]),
        "f": random.normal(rng_f, (B, L)),
    }

    model = FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=2)
    variables = model.init({"params": rng_init, "extra": rng_extra}, **batch)
    assert "params" in variables


def test_flow_matching_call_uses_supplied_t():
    """The placeholder default must not shadow a user-supplied ``t``."""
    B, L, d = 4, 16, 2
    rng = random.key(1)
    rng_init, rng_extra, rng_z, rng_s = random.split(rng, 4)

    z = random.normal(rng_z, (B, L))
    s = random.uniform(rng_s, (L, d))
    conditionals = jnp.array([20.0])

    model = FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=1)
    variables = model.init(
        {"params": rng_init, "extra": rng_extra}, z, conditionals, jnp.zeros((B,)), s=s
    )

    out_zero = model.apply(
        variables, z, conditionals, jnp.zeros((B,)), s=s,
        rngs={"extra": rng_extra},
    )
    out_one = model.apply(
        variables, z, conditionals, jnp.ones((B,)), s=s,
        rngs={"extra": rng_extra},
    )
    # Different t must produce different vector field predictions.
    assert not jnp.allclose(out_zero.f_hat, out_one.f_hat)


def test_flow_matching_decode_runs():
    """The Euler-stepped decoder must produce finite output."""
    B, L, d = 4, 16, 2
    rng = random.key(2)
    rng_init, rng_extra, rng_z, rng_s = random.split(rng, 4)

    z = random.normal(rng_z, (B, L))
    s = random.uniform(rng_s, (L, d))
    conditionals = jnp.array([20.0])

    model = FlowMatchingDeepRV(vf=FlowMatchingVectorField(num_blks=2), n_steps=3)
    variables = model.init(
        {"params": rng_init, "extra": rng_extra}, z=z, conditionals=conditionals, s=s
    )
    out = model.apply(
        variables, z, conditionals, s=s, method="decode",
        rngs={"extra": rng_extra},
    )
    assert out.shape == (B, L)
    assert bool(jnp.all(jnp.isfinite(out)))
