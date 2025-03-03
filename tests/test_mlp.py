from jax import random

from dl4bi.core.attention import MultiHeadAttention
from dl4bi.core.mlp import gMLP, gMLPBlock


def test_gMLP():
    B, L, D = 4, 16, 1
    rng = random.key(42)
    rng_data, rng_init = random.split(rng)
    x = random.normal(rng_data, (B, L, D))
    m = gMLP()
    y, _params = m.init_with_output(rng_init, x)
    assert y.shape == (B, L, D)


def test_aMLP():
    B, L, D = 4, 16, 1
    rng = random.key(42)
    rng_data, rng_init = random.split(rng)
    x = random.normal(rng_data, (B, L, D))
    m = gMLP(blk=gMLPBlock(attn=MultiHeadAttention()))
    y, _params = m.init_with_output(rng_init, x)
    assert y.shape == (B, L, D)
