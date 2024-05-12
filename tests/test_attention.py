import pickle

import jax.numpy as jnp
from jax import random

from dge import (
    AdditiveScorer,
    Attention,
    DotScorer,
    FastSoftmaxAttention,
    MultiheadAttention,
    MultiheadFastSoftmaxAttention,
    MultiplicativeScorer,
)


def test_regular_attention():
    B, L, D = 4, 7, 12
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    data = random.normal(rng_data, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
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
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = MultiheadAttention(scorer, H).init_with_output(
            rng_init, qs, ks, vs, valid_lens
        )
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_fast_softmax_attention():
    B, L, D = 4, 128, 16
    key = random.key(42)
    rng_qkvs, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkvs, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    (ctx_true, _), params_true = Attention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    (ctx_fast, _), params_fast = FastSoftmaxAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    mse = jnp.square(ctx_true - ctx_fast).mean()
    max_error = jnp.max(jnp.abs(ctx_true - ctx_fast))
    assert ctx_true.shape == (B, L, D), "Incorrect context output shape!"
    assert ctx_fast.shape == (B, L, D), "Incorrect context output shape!"
    assert mse < 0.05, "Large MSE error in approximation"
    assert max_error < 2.0, "Large max error in approximation!"
