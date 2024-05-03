import jax.numpy as jnp
from jax import random

from dge import AttentiveNeuralProcess, neural_process_maximum_likelihood_loss


def test_attentive_neural_process_dims():
    B, S, E, N_Z = 4, 10, 16, 32
    key = random.key(42)
    rng_data, rng_init, rng_sample = random.split(key, 3)
    s = jnp.linspace(0, 1.0, S)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_S=1]
    valid_lens = jnp.array([2, 4, 9, 3])
    f = random.normal(rng_data, s.shape)
    (f_mu, f_log_var, _, _), _ = AttentiveNeuralProcess().init_with_output(
        rng_init, rng_sample, s, f, s
    )
    loss = neural_process_maximum_likelihood_loss(f, f_mu, f_log_var)
    assert loss > 0, "Invalid loss!"
