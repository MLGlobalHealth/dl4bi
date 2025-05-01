#!/usr/bin/env -S PYENV_VERSION=torch python3
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
import time

torch.set_default_device("cuda:0")


class BiasedFlexAttention(nn.Module):
    def __init__(self, bias: nn.Module):
        super().__init__()
        self.bias = bias

    def forward(self, qs, ks, vs, qs_s, ks_s):
        def score_mod(score, b, h, q_idx, kv_idx):
            return self.bias(score, b, h, q_idx, kv_idx, qs_s, ks_s)

        return flex_attention(qs, ks, vs, score_mod=score_mod)


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


def sample_batch(
    B: int = 32,
    H: int = 4,
    L: int = 4096,
    D: int = 16,
    D_s: int = 2,
):
    qs, ks, vs = torch.randn(3, B, H, L, D)
    qs_s, ks_s = torch.randn(2, B, L, D_s)
    return {"qs": qs, "ks": ks, "vs": vs, "qs_s": qs_s, "ks_s": ks_s}


if __name__ == "__main__":
    N = 100
    torch.manual_seed(42)
    bias = RBFBias(num_heads=4, num_basis=5)
    attn = BiasedFlexAttention(bias)
    times = []
    b = sample_batch()
    attn(**b)  # precompile
    times = torch.zeros(N)
    for i in range(N):
        b = sample_batch()
        start = time.perf_counter()
        attn(**b)
        stop = time.perf_counter()
        times[i] = stop - start
    print(times.mean())
