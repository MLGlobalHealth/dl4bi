from jax import random

from dl4bi.core.hyper import HyperLoRA, HyperLoRAqkv


def test_hyper_lora_qkv_dims():
    rng = random.key(42)
    rng_data, rng_init = random.split(rng)
    B, L, D = 4, 23, 5
    x, y = random.normal(rng_data, (2, B, L, D))
    m = HyperLoRAqkv()
    for z in [y, y[:, 0]]:  # z: [B, L, D] or [B, D]
        (q, k, v), _params = m.init_with_output(rng_init, x, z)
        assert x.shape == q.shape
        assert x.shape == k.shape
        assert x.shape == v.shape


def test_hyper_lora_dims():
    rng = random.key(42)
    rng_data, rng_init = random.split(rng)
    B, L, D_in, D_out = 4, 23, 16, 32
    x, y = random.normal(rng_data, (2, B, L, D_in))
    m = HyperLoRA(D_out)
    for z in [y, y[:, 0]]:  # z: [B, L, D] or [B, D]
        y, _params = m.init_with_output(rng_init, x, z)
        assert y.shape == (B, L, D_out)
