from time import time

import jax
import jax.numpy as jnp
from jax import random
from jraph import GraphsTuple
from sps.utils import build_grid

from dl4bi.core.attention import (
    Attention,
    DeepKernelAttention,
    FastAttention,
    MultiHeadAttention,
    MultiHeadGraphAttention,
    RBFNetworkBiasedScanAttention,
    ScanAttention,
    TISABiasedScanAttention,
)
from dl4bi.core.utils import mask_from_valid_lens


def test_attention_impl():
    B, H, L, D = 4, 4, 32, 16
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, H, L, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    valid_lens = jnp.array([2, 4, 6, 3])
    mask = mask_from_valid_lens(L, valid_lens)
    (ctx, attn), _ = Attention().init_with_output(rng_init, qs, ks, vs, mask, bias=bias)
    assert ctx.shape == (B, H, L, D), "Incorrect context output shape!"
    assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_multihead_attention_impl():
    B, H, L, D = 4, 4, 32, 64
    key = random.key(42)
    rng_qkv, rng_bias, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    bias = random.normal(rng_bias, (B, H, L, L))
    valid_lens = jnp.array([2, 4, 6, 3])
    mask = mask_from_valid_lens(L, valid_lens)
    (ctx, attn), _ = MultiHeadAttention(attn=Attention(), num_heads=H).init_with_output(
        rng_init, qs, ks, vs, mask, bias=bias
    )
    assert ctx.shape == (B, L, D), "Incorrect context output shape!"
    assert attn.shape == (B, H, L, L), "Incorrect attention output shape!"


def test_multihead_graph_attention_impl():
    B, H, N_c, N_t, D, K = 4, 4, 7, 9, 64, 12
    key = random.key(42)
    (
        rng_nodes,
        rng_edges,
        rng_bias,
        rng_init,
        rng_k_cc,
        rng_k_ct,
    ) = random.split(key, 6)
    nodes = random.normal(rng_nodes, (B * (N_c + N_t), D))
    edges = random.normal(rng_edges, (B * (N_c + N_t) * K,))
    bias = random.normal(rng_bias, (B * (N_c + N_t) * K, H))
    s_cc = random.randint(rng_k_cc, (B, N_c, K), 0, N_c)
    s_cc = s_cc.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_c * K)
    s_ct = random.randint(rng_k_ct, (B, N_t, K), 0, N_c)
    s_ct = s_ct.flatten() + jnp.repeat(jnp.arange(B) * N_c, N_t * K)
    g = GraphsTuple(
        nodes,
        edges,
        senders=jnp.hstack([s_cc, s_ct]),
        receivers=jnp.repeat(jnp.arange(B * (N_c + N_t)), K),
        globals=None,
        n_node=jnp.array([B * (N_c + N_t)]),
        n_edge=jnp.array([B * (N_c + N_t) * K]),
    )
    (ctx, attn), _ = MultiHeadGraphAttention(H).init_with_output(
        rng_init, g, training=False, bias=bias
    )
    assert ctx.shape == (B * (N_c + N_t), D), "Incorrect context output shape!"
    assert attn.shape == (B * (N_c + N_t) * K, H), "Incorrect attention output shape!"


def test_fast_attention_impl():
    B, L, H, D = 4, 128, 4, 16
    key = random.key(42)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, H, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    mask = mask_from_valid_lens(L, valid_lens)
    (ctx_true, _), _ = Attention().init_with_output(rng_init, qs, ks, vs, mask)
    (ctx_fast, _), _ = FastAttention().init_with_output(rng_init, qs, ks, vs, mask)
    mse_fast = jnp.square(ctx_true - ctx_fast).mean()
    max_error_fast = jnp.max(jnp.abs(ctx_true - ctx_fast))
    assert ctx_true.shape == (B, H, L, D), "Full: incorrect context output shape!"
    assert ctx_fast.shape == (B, H, L, D), "Fast: incorrect context output shape!"
    # Source: https://tinyurl.com/google-fast-attn
    assert mse_fast < 0.05, "Fast: Large MSE error in approximation"
    # TODO(danj): this changes based on jax version; is this too high?
    assert max_error_fast < 2.25, "Fast: Large max error in approximation!"


def test_scan_attention_impl():
    B, L, H, D, C = 4, 313, 4, 16, 256
    key = random.key(42)
    rng_qkvs, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkvs, (3, B, H, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    mask = mask_from_valid_lens(L, valid_lens)
    (ctx_true, _), _ = Attention().init_with_output(rng_init, qs, ks, vs, mask)
    (ctx_scan, _), _ = ScanAttention(C, C).init_with_output(rng_init, qs, ks, vs, mask)
    mse_scan = jnp.square(ctx_true - ctx_scan).mean()
    max_error_scan = jnp.max(jnp.abs(ctx_true - ctx_scan))
    assert ctx_true.shape == (B, H, L, D), "Full: incorrect context output shape!"
    assert ctx_scan.shape == (B, H, L, D), "Scan: incorrect context output shape!"
    assert mse_scan < 5e-7, "Scan: Large MSE error in approximation"
    assert max_error_scan < 0.01, "Scan: Large max error in approximation!"


def test_biased_scan_attention_impl():
    B, H, L, D, S, C = 4, 7, 313, 16, 2, 256
    key = random.key(42)
    rng_qkvs, rng_locs, rng_valid, rng_init = random.split(key, 4)
    qs_s, ks_s = random.normal(rng_locs, (2, B, L, S))
    qs, ks, vs = random.normal(rng_qkvs, (3, B, H, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    mask = mask_from_valid_lens(L, valid_lens)
    for cls in [TISABiasedScanAttention, RBFNetworkBiasedScanAttention]:
        (ctx_scan, _), _ = TISABiasedScanAttention(C, C).init_with_output(
            rng_init,
            qs,
            ks,
            vs,
            mask,
            qs_s=qs_s,
            ks_s=ks_s,
        )
        print(cls)
        assert ctx_scan.shape == (B, H, L, D), "Scan: incorrect context output shape!"


def test_fast_softmax_attention_speed():
    B, H, L, D, N = 1, 4, 1024, 16, 5
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    data = random.normal(rng_qkv, (3, B, H, L, D))
    qs, ks, vs = data[0], data[1], data[2]
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    mask = mask_from_valid_lens(L, valid_lens)
    fast_attn = FastAttention()
    _, params = fast_attn.init_with_output(rng_init, qs, ks, vs, mask)
    jit_fast_attn = jax.jit(fast_attn.apply)  # force compile
    jit_fast_attn(params, qs, ks, vs, mask, rngs={"rng_extra": key})
    t_fast_start = time()
    for i in range(N):
        jit_fast_attn(params, qs, ks, vs, mask, rngs={"rng_extra": key})
    t_fast_stop = time()
    t_fast_diff = t_fast_stop - t_fast_start
    del jit_fast_attn, fast_attn, params  # free up memory
    attn = Attention()
    _, p_true = attn.init_with_output(rng_init, qs, ks, vs, mask)
    jit_attn = jax.jit(attn.apply)  # force compile
    jit_attn(p_true, qs, ks, vs, mask)
    t_true_start = time()
    for i in range(N):
        jit_attn(p_true, qs, ks, vs, mask)
    t_true_stop = time()
    t_true_diff = t_true_stop - t_true_start

    assert jnp.isclose(t_fast_diff, t_true_diff, atol=1e-4), "Fast isn't faster!"


def test_scan_attention_speed():
    B, L, H, D, N, C = 5, 1024, 4, 16, 5, 1024
    key = random.key(42)
    rng_qkv, rng_bias, rng_valid, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, H, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    mask = mask_from_valid_lens(L, valid_lens)
    scan_attn = ScanAttention(C, C)
    _, params = scan_attn.init_with_output(rng_init, qs, ks, vs, mask)
    jit_scan_attn = jax.jit(scan_attn.apply)  # force compile
    jit_scan_attn(params, qs, ks, vs, mask, rngs={"rng_extra": key})
    t_scan_start = time()
    for i in range(N):
        jit_scan_attn(params, qs, ks, vs, mask, rngs={"rng_extra": key})
    t_scan_stop = time()
    t_scan_diff = t_scan_stop - t_scan_start
    del jit_scan_attn, scan_attn, params  # free up memory
    attn = Attention()
    _, p_true = attn.init_with_output(rng_init, qs, ks, vs, mask)
    jit_attn = jax.jit(attn.apply)  # force compile
    jit_attn(p_true, qs, ks, vs, mask)
    t_true_start = time()
    for i in range(N):
        jit_attn(p_true, qs, ks, vs, mask)
    t_true_stop = time()
    t_true_diff = t_true_stop - t_true_start

    max_t, factor = 5e-5, 1.05
    # NOTE: can use the following assert for benchmarking
    # assert t_scan_diff < max_t, f"Scan takes longer than {max_t}s!"
    assert t_scan_diff < factor * t_true_diff, f"Scan is more than {factor}x slower!"


# NOTE: this is expected to be slower since TISA is calculating bias,
# and regular attention is not using any bias
def test_biased_scan_attention_speed():
    B, H, L, D, S, N, C = 5, 4, 1024, 16, 2, 5, 1024
    key = random.key(42)
    rng_qkv, rng_s, rng_valid, rng_init = random.split(key, 4)
    qs, ks, vs = random.normal(rng_qkv, (3, B, H, L, D))
    qs_s, ks_s = random.normal(rng_s, (2, B, L, S))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L)
    mask = mask_from_valid_lens(L, valid_lens)
    for cls in [TISABiasedScanAttention, RBFNetworkBiasedScanAttention]:
        scan_attn = cls(C, C)
        _, params = scan_attn.init_with_output(
            rng_init, qs, ks, vs, mask, qs_s=qs_s, ks_s=ks_s
        )
        jit_scan_attn = jax.jit(scan_attn.apply)  # force compile
        jit_scan_attn(
            params,
            qs,
            ks,
            vs,
            mask,
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
                mask,
                qs_s=qs_s,
                ks_s=ks_s,
                rngs={"rng_extra": key},
            )
        t_scan_stop = time()
        t_scan_diff = t_scan_stop - t_scan_start
        del jit_scan_attn, scan_attn, params  # free up memory
        attn = Attention()
        _, p_true = attn.init_with_output(rng_init, qs, ks, vs, mask)
        jit_attn = jax.jit(attn.apply)  # force compile
        jit_attn(p_true, qs, ks, vs, mask)
        t_true_start = time()
        for i in range(N):
            jit_attn(p_true, qs, ks, vs, mask)
        t_true_stop = time()
        t_true_diff = t_true_stop - t_true_start

        max_t, factor = 5e-5, 1.1
        # NOTE: can use the following assert for benchmarking
        # assert t_scan_diff < max_t, f"Scan takes longer than {max_t}s!"
        assert (
            t_scan_diff < factor * t_true_diff
        ), f"Scan is more than {factor}x slower!"


def test_fast_softmax_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, L_ctx, L_test, L_init, H, D = 1, 110000, 50000, 3, 4, 16
    key = random.key(42)
    rng_init, rng_qs, rng_kvs = random.split(key, 3)
    x = random.normal(rng_init, (B, H, L_init, D))
    qs = random.normal(rng_qs, (B, H, L_test, D))
    kvs = random.normal(rng_kvs, (B, H, L_ctx, D))

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
    qs = random.normal(rng_qs, (B, H, L_test, D))
    kvs = random.normal(rng_kvs, (B, H, L_ctx, D))

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


def test_biased_scan_attention_scale():
    # L_ctx, L_test = 105569, 44431  # Case Study for Large Spatial Data, Heaton et al
    B, H, L_ctx, L_test, D, S = 1, 4, 110000, 50000, 16, 2
    key = random.key(42)
    rng_init, rng_qs_s, rng_ks_s, rng_qs, rng_kvs = random.split(key, 5)
    qs_s = random.normal(rng_qs_s, (B, L_test, S))
    ks_s = random.normal(rng_ks_s, (B, L_ctx, S))
    qs = random.normal(rng_qs, (B, H, L_test, D))
    kvs = random.normal(rng_kvs, (B, H, L_ctx, D))

    for cls in [TISABiasedScanAttention, RBFNetworkBiasedScanAttention]:
        scan_attn = cls()
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


def test_deep_kernel_attention_impl():
    B, L, D = 4, 128, 64
    key = random.key(42)
    s = jnp.repeat(build_grid()[None, ...], B, axis=0)
    rng_qkv, rng_valid, rng_init = random.split(key, 3)
    qs, ks, vs = random.normal(rng_qkv, (3, B, L, D))
    valid_lens = random.randint(rng_valid, (B,), 0, maxval=L, dtype=jnp.int32)
    mask = mask_from_valid_lens(L, valid_lens)
    (ctx, _), _ = DeepKernelAttention().init_with_output(
        rng_init, qs, ks, vs, mask, qs_s=s, ks_s=s
    )
    assert jnp.isfinite(ctx).all(), "KernelAttention produced non-finite values!"
    assert ctx.shape == (B, L, D), "Incorrect context output shape!"
