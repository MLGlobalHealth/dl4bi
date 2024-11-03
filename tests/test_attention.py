from time import time

import jax
import jax.numpy as jnp
from jax import random
from jaxlib.xla_client import XlaRuntimeError

from dl4bi.core import (
    MLP,
    AdditiveScorer,
    Attention,
    DotScorer,
    FastAttention,
    FusedAttention,
    KernelAttention,
    MultiHeadAttention,
    MultiKernelAttention,
    MultiplicativeScorer,
    exponential_scorer,
    rbf_scorer,
)


def test_vanilla_attention():
    B, L, H, D = 4, 7, 4, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = Attention(scorer).init_with_output(
            rng_init, qs, ks, vs, bias, valid_lens
        )
        assert ctx.shape == (B, L, H, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_multihead_attention():
    B, H, L, D = 4, 4, 7, 64
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    data = random.normal(rng_qkv, (3, B, L, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = MultiHeadAttention(
            attn=Attention(scorer), num_heads=H
        ).init_with_output(rng_init, qs, ks, vs, bias, valid_lens)
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_fast_attention():
    B, L, H, D = 4, 128, 4, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    bias = None  # not yet supported
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_true, _), _ = Attention().init_with_output(
        rng_init, qs, ks, vs, bias, valid_lens
    )
    (ctx_fast, _), _ = FastAttention().init_with_output(
        rng_init, qs, ks, vs, bias, valid_lens
    )
    mse_fast = jnp.square(ctx_true - ctx_fast).mean()
    max_error_fast = jnp.max(jnp.abs(ctx_true - ctx_fast))
    assert ctx_true.shape == (B, L, H, D), "Full: incorrect context output shape!"
    assert ctx_fast.shape == (B, L, H, D), "Fast: incorrect context output shape!"
    # Source: https://tinyurl.com/google-fast-attn
    assert mse_fast < 0.05, "Fast: Large MSE error in approximation"
    assert max_error_fast < 2.0, "Fast: Large max error in approximation!"


def test_fused_attention():
    B, L, H, D = 4, 128, 4, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_true, _), _ = Attention().init_with_output(
        rng_init, qs, ks, vs, bias, valid_lens
    )
    (ctx_fused, _), _ = FusedAttention().init_with_output(
        rng_init, qs, ks, vs, bias, valid_lens
    )
    mse_fused = jnp.square(ctx_true - ctx_fused).mean()
    max_error_fused = jnp.max(jnp.abs(ctx_true - ctx_fused))
    assert ctx_true.shape == (B, L, H, D), "Full: incorrect context output shape!"
    assert ctx_fused.shape == (B, L, H, D), "Fused: incorrect context output shape!"
    # TODO(danj): is this expected?
    assert mse_fused < 0.01, "Fused: Large MSE error in approximation"
    assert max_error_fused < 1.0, "Fused: Large max error in approximation!"


def test_fast_softmax_attention_speed():
    B, L, H, D = 1, 32768, 4, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    bias = None  # not yet supported
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    fast_attn = FastAttention()
    _, params = fast_attn.init_with_output(rng_init, qs, ks, vs, bias, valid_lens)
    jit_fast_attn = jax.jit(fast_attn.apply)
    t_fast_start = time()
    for i in range(3):
        jit_fast_attn(params, qs, ks, vs, bias, valid_lens, rngs={"rng_extra": key})
    t_fast_stop = time()
    t_fast_diff = t_fast_stop - t_fast_start
    del jit_fast_attn, fast_attn, params  # free up memory

    try:
        attn = Attention()
        _, p_true = attn.init_with_output(rng_init, qs, ks, vs, bias, valid_lens)
        jit_attn = jax.jit(attn.apply)
        t_true_start = time()
        for i in range(3):
            jit_attn(p_true, qs, ks, vs, bias, valid_lens)
        t_true_stop = time()
        t_true_diff = t_true_stop - t_true_start
    except XlaRuntimeError:  # OOM
        t_true_diff = 1e6

    assert t_fast_diff < t_true_diff, "Fast isn't faster!"


def test_fast_softmax_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, L_ctx, L_test, L_init, H, D = 1, 110000, 50000, 3, 4, 16
    key = random.key(42)
    rng_init, rng_qs, rng_kvs = random.split(key, 3)
    x = random.normal(rng_init, (B, L_init, H, D))
    qs = random.normal(rng_qs, (B, L_test, H, D))
    kvs = random.normal(rng_kvs, (B, L_ctx, H, D))
    bias = None  # not yet supported

    fast_attn = FastAttention()
    (ctx_fast_init, _), params = fast_attn.init_with_output(rng_init, x, x, x, bias)
    jit_fast_attn = jax.jit(
        lambda qs, ks, vs: fast_attn.apply(
            params, qs, ks, vs, bias, rngs={"rng_extra": key}
        )
    )
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        ctx_fast, _ = jit_fast_attn(qs, kvs, kvs)

    assert jnp.isfinite(ctx_fast_init).all(), "Non-finite values produced!"
    assert jnp.isfinite(ctx_fast).all(), "Non-finite values produced!"


def test_kernel_attention():
    B, L, D = 4, 128, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkv, (3, B, L, D))
    bias = None  # not yet supported
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    for kernel in [rbf_scorer, exponential_scorer]:
        (ctx, _), _ = KernelAttention(kernel, proj_out=MLP([D])).init_with_output(
            rng_init, qs, ks, vs, bias, valid_lens
        )
        assert jnp.isfinite(ctx).all(), "KernelAttention produced non-finite values!"
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"


def test_multikernel_attention():
    B, L, D = 4, 128, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    data = random.normal(rng_qkv, (3, B, L, D))
    bias = None  # not yet supported
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    kernels = [KernelAttention(k) for k in [rbf_scorer, exponential_scorer]]
    (ctx, _), _ = MultiKernelAttention(kernels, proj_out=MLP([D])).init_with_output(
        rng_init, qs, ks, vs, bias, valid_lens
    )
    assert jnp.isfinite(ctx).all(), "MultikernelAttention produced non-finite values!"
    assert ctx.shape == (B, L, D), "Incorrect context output shape!"
