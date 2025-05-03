#!/usr/bin/env -S PYENV_VERSION=torch python3
import argparse
import sys
import time

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

torch.set_default_device("cuda:0")

# NOTE: cannot get this to scale past ~L=1600
# issue: https://github.com/pytorch/pytorch/issues/152593


def main(seed: int, N: int, B: int, H: int, L: int, D: int, D_s: int):
    torch.manual_seed(seed)
    bias = RBFBias(num_heads=4, num_basis=5)
    kernel_options = {}
    # kernel_options = { # NOTE: doesn't help
    #     "BLOCK_M": 64,
    #     "BLOCK_N": 64,
    #     "BLOCK_M1": 32,
    #     "BLOCK_N1": 64,
    #     "BLOCK_M2": 64,
    #     "BLOCK_N2": 32,
    # }  # https://github.com/pytorch/pytorch/issues/133254
    attn = BiasedFlexAttention(bias, kernel_options)
    # attn = torch.compile(attn, dynamic=False)  # NOTE: fails
    b = sample_batch(B, H, L, D, D_s)
    attn(**b)  # precompile flex_attention (?)
    torch.cuda.synchronize()
    times = torch.zeros(N)
    for i in range(N):
        b = sample_batch(B, H, L, D, D_s)
        torch.cuda.synchronize()
        start = time.perf_counter()
        attn(**b)
        torch.cuda.synchronize()
        stop = time.perf_counter()
        times[i] = stop - start
    print(
        f"[B={B} H={H} L={L} D={D} D_s={D_s}]: {times.mean():0.6f}±{times.std():0.6f}s"
    )


class RBFBias(nn.Module):
    def __init__(self, num_heads, num_basis):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(num_heads, num_basis))
        self.beta = nn.Parameter(torch.randn(num_heads, num_basis))

    def forward(self, score, b, h, q_idx, kv_idx, qs_s, ks_s):
        q_s = qs_s[b, q_idx]
        k_s = ks_s[b, kv_idx]
        d_sq = torch.square(q_s - k_s).sum()
        alpha, beta = self.alpha[h], self.beta[h]
        d_rbf = alpha * torch.exp(-beta * d_sq)
        return score + d_rbf.sum()


class BiasedFlexAttention(nn.Module):
    def __init__(self, bias: nn.Module, kernel_options: dict = {}):
        super().__init__()
        self.bias = bias
        self.kernel_options = kernel_options

    def forward(self, qs, ks, vs, qs_s, ks_s):
        def score_mod(score, b, h, q_idx, kv_idx):
            return self.bias(score, b, h, q_idx, kv_idx, qs_s, ks_s)

        return flex_attention(
            qs,
            ks,
            vs,
            score_mod=score_mod,
            kernel_options=self.kernel_options,
        )


def sample_batch(B: int, H: int, L: int, D: int, D_s: int):
    qs, ks, vs = torch.randn(3, B, H, L, D)
    qs_s, ks_s = torch.randn(2, B, L, D_s)
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
