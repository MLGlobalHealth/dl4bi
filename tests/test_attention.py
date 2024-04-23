import jax.numpy as jnp
from jax import random

from dge import AdditiveScorer, Attention, DotScorer, MultiheadAttention


def test_regular_attention():
    B, L, D = 4, 7, 12
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    data = random.normal(rng_data, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), DotScorer()]:
        (ctx, attn), _ = Attention(scorer).init_with_output(
            rng_init, qs, ks, vs, valid_lens
        )
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, L, L), "Incorrect attention output shape!"


def test_multihead_attention():
    B, H, L, D = 4, 4, 7, 12
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    data = random.normal(rng_data, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), DotScorer()]:
        (ctx, attn), _ = MultiheadAttention(Attention(scorer)).init_with_output(
            rng_init, qs, ks, vs, valid_lens
        )
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"
