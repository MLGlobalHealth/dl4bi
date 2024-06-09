import jax.numpy as jnp
from jax import random

from dge.meta_regression import NP


def test_vanilla_neural_process():
    B, L, d_ffn, d_z, n_z = 4, 10, 128, 128, 1
    key = random.key(42)
    rng_data, rng_params, rng_dropout, rng_latent_z = random.split(key, 4)
    s = jnp.linspace(0, 1.0, L)
    s = jnp.repeat(s[None, :, None], B, axis=0)  # [B, S, D_s=1]
    valid_lens = jnp.array([2, 4, 9, 3])
    f = random.normal(rng_data, s.shape)
    (f_mu, f_log_var, z_mu_ctx, z_std_ctx), params = NP(
        d_ffn, d_z, n_z
    ).init_with_output(
        {"params": rng_params, "dropout": rng_dropout, "latent_z": rng_latent_z},
        s_ctx=s,
        f_ctx=f,
        s_test=s,
        valid_lens_ctx=valid_lens,
        valid_lens_test=valid_lens,
        training=True,
    )
    assert f_mu.shape == (B, L, 1)
