from time import time

import jax
import jax.numpy as jnp
from jax import random

from dl4bi.core import (
    MLP,
    AdditiveScorer,
    Attention,
    DeepKernelAttention,
    DistanceBiasedFastAttention,
    DotScorer,
    FastAttention,
    FusedAttention,
    KernelAttention,
    MultiHeadAttention,
    MultiKernelAttention,
    MultiplicativeScorer,
    ScanAttention,
    SpatioTemporalMLPAttention,
    TISABiasedScanAttention,
    exponential_scorer,
    rbf_scorer,
)


def test_vanilla_attention_impl():
    B, L, H, D = 4, 7, 4, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, H, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = Attention(scorer).init_with_output(
            rng_init, qs, ks, vs, valid_lens, bias=bias
        )
        assert ctx.shape == (B, L, H, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_multihead_attention_impl():
    B, H, L, D = 4, 4, 7, 64
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    valid_lens = jnp.array([2, 4, 6, 3])
    for scorer in [AdditiveScorer(), MultiplicativeScorer(), DotScorer()]:
        (ctx, attn), _ = MultiHeadAttention(
            attn=Attention(scorer), num_heads=H
        ).init_with_output(rng_init, qs, ks, vs, valid_lens, bias=bias)
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_spatiotemporal_mlp_attention_impl():
    B, L, D, S, T = 4, 7, 64, 2, 1
    key = random.key(42)
    rng_qkv, rng_s, rng_t, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    qs_s, ks_s = random.normal(rng_s, (2, B, L, S))
    ks_t = random.normal(rng_t, (B, L, T))
    qs_t = ks_t + 1  # queries are temporally later than keys
    valid_lens = jnp.array([2, 4, 6, 3])
    for vnode in [None, jnp.ones((B, 1, D))]:
        (ctx, attn), _ = SpatioTemporalMLPAttention().init_with_output(
            rng_init,
            qs,
            ks,
            vs,
            valid_lens,
            qs_s=qs_s,
            ks_s=ks_s,
            qs_t=qs_t,
            ks_t=ks_t,
            vnode=vnode,
        )
        assert jnp.isfinite(ctx).all(), "Produced non-finite output!"
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"
        assert attn.shape == (B, L, L), "Incorrect attention output shape!"


def test_fast_attention_impl():
    B, L, H, D = 4, 128, 4, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, H, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_true, _), _ = Attention().init_with_output(rng_init, qs, ks, vs, valid_lens)
    (ctx_fast, _), _ = FastAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    mse_fast = jnp.square(ctx_true - ctx_fast).mean()
    max_error_fast = jnp.max(jnp.abs(ctx_true - ctx_fast))
    assert ctx_true.shape == (B, L, H, D), "Full: incorrect context output shape!"
    assert ctx_fast.shape == (B, L, H, D), "Fast: incorrect context output shape!"
    # Source: https://tinyurl.com/google-fast-attn
    assert mse_fast < 0.05, "Fast: Large MSE error in approximation"
    assert max_error_fast < 2.0, "Fast: Large max error in approximation!"


def test_distance_biased_fast_attention_impl():
    B, L, H, D, S = 4, 128, 4, 16, 2
    key = random.key(42)
    rng_qkv, rng_s, rng_valid, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, H, D))
    qs_s, ks_s = random.normal(rng_s, (2, B, L, S))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_fast, _), _ = DistanceBiasedFastAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens, qs_s=qs_s, ks_s=ks_s
    )
    assert ctx_fast.shape == (
        B,
        L,
        H,
        D,
    ), "DistanceBiasedFast: incorrect context output shape!"


def test_scan_attention_impl():
    B, L, H, D, C = 4, 313, 4, 16, 256
    key = random.key(42)
    rng_qkvs, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkvs, (3, B, L, H, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_true, _), _ = Attention().init_with_output(rng_init, qs, ks, vs, valid_lens)
    (ctx_scan, _), _ = ScanAttention(C, C).init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    mse_scan = jnp.square(ctx_true - ctx_scan).mean()
    max_error_scan = jnp.max(jnp.abs(ctx_true - ctx_scan))
    assert ctx_true.shape == (B, L, H, D), "Full: incorrect context output shape!"
    assert ctx_scan.shape == (B, L, H, D), "Scan: incorrect context output shape!"
    assert mse_scan < 1e-7, "Scan: Large MSE error in approximation"
    assert max_error_scan < 0.01, "Scan: Large max error in approximation!"


def test_tisa_biased_scan_attention_impl():
    B, L, H, D, S, C = 4, 313, 4, 16, 2, 256
    key = random.key(42)
    rng_qkvs, rng_locs, rng_valid, rng_init = random.split(key, 4)
    qs_s, ks_s = random.normal(rng_locs, (2, B, L, S))
    qs, ks, vs = random.normal(rng_qkvs, (3, B, L, H, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_scan, _), _ = TISABiasedScanAttention(C, C).init_with_output(
        rng_init,
        qs,
        ks,
        vs,
        valid_lens,
        qs_s=qs_s,
        ks_s=ks_s,
    )
    assert ctx_scan.shape == (B, L, H, D), "Scan: incorrect context output shape!"


def test_fused_attention_impl():
    B, L, H, D = 4, 128, 4, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx_true, _), _ = Attention().init_with_output(
        rng_init, qs, ks, vs, valid_lens, bias=bias
    )
    (ctx_fused, _), _ = FusedAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens, bias=bias
    )
    mse_fused = jnp.square(ctx_true - ctx_fused).mean()
    max_error_fused = jnp.max(jnp.abs(ctx_true - ctx_fused))
    assert ctx_true.shape == (B, L, H, D), "Full: incorrect context output shape!"
    assert ctx_fused.shape == (B, L, H, D), "Fused: incorrect context output shape!"
    # TODO(danj): is this expected?
    assert mse_fused < 0.01, "Fused: Large MSE error in approximation"
    assert max_error_fused < 1.0, "Fused: Large max error in approximation!"


def test_fast_softmax_attention_speed():
    B, L, H, D, N = 1, 1024, 4, 16, 5
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkv, (3, B, L, H, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    fast_attn = FastAttention()
    _, params = fast_attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_fast_attn = jax.jit(fast_attn.apply)  # force compile
    jit_fast_attn(params, qs, ks, vs, valid_lens, rngs={"rng_extra": key})
    t_fast_start = time()
    for i in range(N):
        jit_fast_attn(params, qs, ks, vs, valid_lens, rngs={"rng_extra": key})
    t_fast_stop = time()
    t_fast_diff = t_fast_stop - t_fast_start
    del jit_fast_attn, fast_attn, params  # free up memory
    attn = Attention()
    _, p_true = attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_attn = jax.jit(attn.apply)  # force compile
    jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_start = time()
    for i in range(N):
        jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_stop = time()
    t_true_diff = t_true_stop - t_true_start

    assert jnp.isclose(t_fast_diff, t_true_diff, atol=1e-4), "Fast isn't faster!"


def test_scan_attention_speed():
    B, L, H, D, N, C = 5, 1024, 4, 16, 5, 1024
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, H, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    scan_attn = ScanAttention(C, C)
    _, params = scan_attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_scan_attn = jax.jit(scan_attn.apply)  # force compile
    jit_scan_attn(params, qs, ks, vs, valid_lens, rngs={"rng_extra": key})
    t_scan_start = time()
    for i in range(N):
        jit_scan_attn(params, qs, ks, vs, valid_lens, rngs={"rng_extra": key})
    t_scan_stop = time()
    t_scan_diff = t_scan_stop - t_scan_start
    del jit_scan_attn, scan_attn, params  # free up memory
    attn = Attention()
    _, p_true = attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_attn = jax.jit(attn.apply)  # force compile
    jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_start = time()
    for i in range(N):
        jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_stop = time()
    t_true_diff = t_true_stop - t_true_start

    max_t, factor = 5e-5, 1.1
    # NOTE: can use the following assert for benchmarking
    # assert t_scan_diff < max_t, f"Scan takes longer than {max_t}s!"
    assert t_scan_diff < factor * t_true_diff, f"Scan is more than {factor}x slower!"


# NOTE: this is expected to be slower since TISA is calculating bias,
# and regular attention is not using any bias
def test_tisa_biased_scan_attention_speed():
    B, L, H, D, S, N, C = 5, 1024, 4, 16, 2, 5, 1024
    key = random.key(42)
    rng_qkv, rng_s, rng_valid, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, H, D))
    qs_s, ks_s = random.normal(rng_s, (2, B, L, S))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    scan_attn = TISABiasedScanAttention(C, C)
    _, params = scan_attn.init_with_output(
        rng_init, qs, ks, vs, valid_lens, qs_s=qs_s, ks_s=ks_s
    )
    jit_scan_attn = jax.jit(scan_attn.apply)  # force compile
    jit_scan_attn(
        params,
        qs,
        ks,
        vs,
        valid_lens,
        qs_s=qs_s,
        ks_s=ks_s,
        rngs={"rng_extra": key},
    )
    t_scan_start = time()
    for i in range(N):
        jit_scan_attn(
            params,
            qs,
            ks,
            vs,
            valid_lens,
            qs_s=qs_s,
            ks_s=ks_s,
            rngs={"rng_extra": key},
        )
    t_scan_stop = time()
    t_scan_diff = t_scan_stop - t_scan_start
    del jit_scan_attn, scan_attn, params  # free up memory
    attn = Attention()
    _, p_true = attn.init_with_output(rng_init, qs, ks, vs, valid_lens)
    jit_attn = jax.jit(attn.apply)  # force compile
    jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_start = time()
    for i in range(N):
        jit_attn(p_true, qs, ks, vs, valid_lens)
    t_true_stop = time()
    t_true_diff = t_true_stop - t_true_start

    max_t, factor = 5e-5, 1.2
    # NOTE: can use the following assert for benchmarking
    # assert t_scan_diff < max_t, f"Scan takes longer than {max_t}s!"
    assert t_scan_diff < factor * t_true_diff, f"Scan is more than {factor}x slower!"


def test_fast_softmax_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, L_ctx, L_test, L_init, H, D = 1, 110000, 50000, 3, 4, 16
    key = random.key(42)
    rng_init, rng_qs, rng_kvs = random.split(key, 3)
    x = random.normal(rng_init, (B, L_init, H, D))
    qs = random.normal(rng_qs, (B, L_test, H, D))
    kvs = random.normal(rng_kvs, (B, L_ctx, H, D))

    fast_attn = FastAttention()
    (ctx_fast_init, _), params = fast_attn.init_with_output(rng_init, x, x, x)
    jit_fast_attn = jax.jit(
        lambda qs, ks, vs: fast_attn.apply(params, qs, ks, vs, rngs={"rng_extra": key})
    )
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        ctx_fast, _ = jit_fast_attn(qs, kvs, kvs)

    assert jnp.isfinite(ctx_fast_init).all(), "Non-finite values produced!"
    assert jnp.isfinite(ctx_fast).all(), "Non-finite values produced!"


def test_scan_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, L_ctx, L_test, H, D = 1, 110000, 50000, 4, 16
    key = random.key(42)
    rng_init, rng_qs, rng_kvs = random.split(key, 3)
    qs = random.normal(rng_qs, (B, L_test, H, D))
    kvs = random.normal(rng_kvs, (B, L_ctx, H, D))

    scan_attn = ScanAttention()
    (ctx_scan_init, _), params = scan_attn.init_with_output(rng_init, qs, kvs, kvs)
    jit_scan_attn = jax.jit(
        lambda qs, ks, vs: scan_attn.apply(params, qs, ks, vs, rngs={"rng_extra": key})
    )
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        ctx_scan, _ = jit_scan_attn(qs, kvs, kvs)

    assert jnp.isfinite(ctx_scan_init).all(), "Non-finite values produced!"
    assert jnp.isfinite(ctx_scan).all(), "Non-finite values produced!"


def test_tisa_biased_scan_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, L_ctx, L_test, H, D, S = 1, 110000, 50000, 4, 16, 2
    key = random.key(42)
    rng_init, rng_qs_s, rng_ks_s, rng_qs, rng_kvs = random.split(key, 5)
    qs_s = random.normal(rng_qs_s, (B, L_test, S))
    ks_s = random.normal(rng_ks_s, (B, L_ctx, S))
    qs = random.normal(rng_qs, (B, L_test, H, D))
    kvs = random.normal(rng_kvs, (B, L_ctx, H, D))

    scan_attn = TISABiasedScanAttention()
    (ctx_scan_init, _), params = scan_attn.init_with_output(
        rng_init, qs, kvs, kvs, qs_s=qs_s, ks_s=ks_s
    )
    jit_scan_attn = jax.jit(
        lambda qs, ks, vs: scan_attn.apply(
            params, qs, ks, vs, qs_s=qs_s, ks_s=ks_s, rngs={"rng_extra": key}
        )
    )
    # to view results: tensorboard --logdir /tmp/tensorboard/
    with jax.profiler.trace("/tmp/tensorboard"):
        ctx_scan, _ = jit_scan_attn(qs, kvs, kvs)

    assert jnp.isfinite(ctx_scan_init).all(), "Non-finite values produced!"
    assert jnp.isfinite(ctx_scan).all(), "Non-finite values produced!"


def test_kernel_attention_impl():
    B, L, D = 4, 128, 64
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    for kernel in [rbf_scorer, exponential_scorer]:
        (ctx, _), _ = KernelAttention(kernel).init_with_output(
            rng_init, qs, ks, vs, valid_lens
        )
        assert jnp.isfinite(ctx).all(), "KernelAttention produced non-finite values!"
        assert ctx.shape == (B, L, D), "Incorrect context output shape!"


def test_deep_kernel_attention_impl():
    B, L, D = 4, 128, 64
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    (ctx, _), _ = DeepKernelAttention().init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    assert jnp.isfinite(ctx).all(), "KernelAttention produced non-finite values!"
    assert ctx.shape == (B, L, D), "Incorrect context output shape!"


def test_multikernel_attention():
    B, L, D = 4, 128, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    kernels = [KernelAttention(k) for k in [rbf_scorer, exponential_scorer]]
    (ctx, _), _ = MultiKernelAttention(kernels, proj_out=MLP([D])).init_with_output(
        rng_init, qs, ks, vs, valid_lens
    )
    assert jnp.isfinite(ctx).all(), "MultikernelAttention produced non-finite values!"
    assert ctx.shape == (B, L, D), "Incorrect context output shape!"


if __name__ == "__main__":
    test_biased_fast_attention_impl()
