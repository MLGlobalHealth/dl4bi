#!/usr/bin/env python3
import argparse
import sys
import time

import jax
import jax.numpy as jnp
from jax import jit, random

from dl4bi.core.attention import BiasedScanAttention
from dl4bi.core.bias import Bias


def main(seed: int, N: int, B: int, H: int, L: int, D: int, D_s: int):
    rng = random.key(seed)
    bias = Bias.build_rbf_network_bias(num_heads=H, num_basis=5)
    attn = BiasedScanAttention(s_bias=bias)
    b = sample_batch(rng, B, H, L, D, D_s)
    params = attn.init(rng, **b)
    jit_attn = jit(
        lambda qs, ks, vs, qs_s, ks_s: attn.apply(
            params, qs, ks, vs, qs_s=qs_s, ks_s=qs_s
        )
    )
    out, *_ = jit_attn(**b)  # precompile
    out.block_until_ready()
    times = jnp.zeros((N,))
    for i in range(N):
        rng_i, rng = random.split(rng)
        b = sample_batch(rng_i, B, H, L, D, D_s)
        b["qs"].block_until_ready()
        start = time.perf_counter()
        out, *_ = jit_attn(**b)
        out.block_until_ready()
        stop = time.perf_counter()
        times = times.at[i].set(stop - start)
    print(
        f"[B={B} H={H} L={L} D={D} D_s={D_s}]: {times.mean():0.6f}±{times.std():0.6f}s"
    )


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


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=42,
        help="Seed.",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=1000,
        help="Number of trials.",
    )
    parser.add_argument(
        "-B",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "-L",
        type=int,
        default=1664,  # 1024 + 512 + 128
        help="Length of sequence to test.",
    )
    parser.add_argument(
        "-H",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "-D",
        type=int,
        default=16,
        help="Embedding dim per head.",
    )
    parser.add_argument(
        "-D_s",
        type=int,
        default=2,
        help="Spatial dimension.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(**vars(args))
