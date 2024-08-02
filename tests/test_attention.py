from time import time

import jax
import jax.numpy as jnp
from jax import random
from jaxlib.xla_client import XlaRuntimeError

from dsp.core import (
    AdditiveScorer,
    Attention,
    DotScorer,
    FastAttention,
    MultiheadAttention,
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
    B, H, L, D = 4, 4, 7, 64
    key = random.key(42)
    rng_data, rng_init = random.split(key)
    data = random.normal(rng_data, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = MultiheadAttention(
            scorer=scorer, num_heads=H
        ).init_with_output(rng_init, qs, ks, vs, valid_lens)
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_fast_attention():
    B, L, D = 4, 128, 16
    key = random.key(42)
    rng_qkvs, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkvs, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    (ctx_true, _), p_true = Attention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    (ctx_fast, _), p_fast = FastAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    mse = jnp.square(ctx_true - ctx_fast).mean()
    max_error = jnp.max(jnp.abs(ctx_true - ctx_fast))
    assert ctx_true.shape == (B, L, D), "Incorrect context output shape!"
    assert ctx_fast.shape == (B, L, D), "Incorrect context output shape!"
    # Source: https://tinyurl.com/google-fast-attn
    assert mse < 0.03, "Large MSE error in approximation"
    assert max_error < 2.0, "Large max error in approximation!"


def test_fast_softmax_attention_speed():
    B, L, D = 1, 20480, 16
    key = random.key(42)
    rng_qkvs, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkvs, (3, B, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)

    fast_attn = FastAttention()
    _, p_fast = fast_attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_fast_attn = jax.jit(fast_attn.apply)
    t_fast_start = time()
    for i in range(3):
        jit_fast_attn(p_fast, qs, ks, vs, valid_lens)
    t_fast_stop = time()
    t_fast_diff = t_fast_stop - t_fast_start
    del jit_fast_attn, fast_attn, p_fast  # free up memory

    try:
        attn = Attention()
        _, p_true = attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
        jit_attn = jax.jit(attn.apply)
        t_true_start = time()
        for i in range(3):
            jit_attn(p_true, qs, ks, vs, valid_lens)
        t_true_stop = time()
        t_true_diff = t_true_stop - t_true_start
    except XlaRuntimeError:  # OOM
        t_true_diff = 1e6

    assert t_fast_diff < t_true_diff, "Fast isn't faster!"


def test_fast_softmax_attention_scale():
    B, L, L_init, D = 1, 32768, 3, 16
    key = random.key(42)
    rng_qkvs_init, rng_qkvs, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkvs, (3, B, L, D))
    data_init = random.normal(rng_qkvs_init, (3, B, L_init, D))
    qs, ks, vs = data[0], data[1], data[2]
    qs_init, ks_init, vs_init = data_init[0], data_init[1], data_init[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    valid_lens_init = random.randint(rng_valid, (B,), 0, maxval=L_init)

    fast_attn = FastAttention()
    (ctx_fast_init, _), p_fast = fast_attn.init_with_output(
        rng_init, qs_init, ks_init, vs_init, valid_lens_init
    )

    jit_fast_attn = jax.jit(fast_attn.apply)
    ctx_fast, _ = jit_fast_attn(p_fast, qs, ks, vs, valid_lens)

    # NOTE: regular attention OOMs around L=20480

    assert not jnp.isnan(ctx_fast_init).any(), "NaNs produced during initialization!"
    assert not jnp.isnan(ctx_fast).any(), "NaNs produced!"
