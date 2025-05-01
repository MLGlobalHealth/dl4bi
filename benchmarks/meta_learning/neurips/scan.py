#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from dl4bi.core.attention import BiasedScanAttention
from dl4bi.core.bias import Bias
from jax import random, jit
import time


def sample_batch(
    rng: jax.Array,
    B: int = 32,
    H: int = 4,
    L: int = 4096,
    D: int = 16,
    D_s: int = 2,
):
    qs, ks, vs = random.normal(rng, (3, B, H, L, D))
    qs_s, ks_s = random.normal(rng, (2, B, L, D_s))
    return {"qs": qs, "ks": ks, "vs": vs, "qs_s": qs_s, "ks_s": ks_s}


if __name__ == "__main__":
    N = 100
    rng = random.key(42)
    bias = Bias.build_rbf_network_bias(num_heads=4, num_basis=5)
    attn = BiasedScanAttention(s_bias=bias)
    b = sample_batch(rng)
    params = attn.init(rng, **b)
    jit_attn = jit(
        lambda qs, ks, vs, qs_s, ks_s: attn.apply(
            params, qs, ks, vs, qs_s=qs_s, ks_s=qs_s
        )
    )
    jit_attn(**b)  # precompile
    times = jnp.zeros((N,))
    for i in range(N):
        rng_i, rng = random.split(rng)
        b = sample_batch(rng_i)
        start = time.perf_counter()
        jit_attn(**b)
        stop = time.perf_counter()
        times = times.at[i].set(stop - start)
    print(times.mean())
