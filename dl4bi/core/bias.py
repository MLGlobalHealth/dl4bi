from typing import Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import l2_dist
import operator as op


class _BiasOp(nn.Module):
    bias_1: nn.Module
    bias_2: nn.Module
    op: op.add

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None):
        return self.bias_1(x) + self.bias_2(x)


class _BiasMul(nn.Module):
    bias_1: nn.Module
    bias_2: nn.Module

    @nn.compact
    def __call__(self, x: jax.Array, mask: Optional[jax.Array] = None):
        return self.bias_1(x) + self.bias_2(x)


class Bias(nn.Module):
    def __add__(self, other):
        return _BiasAdd(self, other)

    def __mul__(self, other):
        return _BiasMultiply(self, other)


class DistanceBias(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array):
        d = jnp.repeat(d[:, None, ...], self.num_heads, axis=1)
        a = self.param("a", init.constant(-1), (1, self.num_heads, 1, 1))
        return a * d  # [B, H, Q, K]


class RBFNetworkBias(nn.Module):
    num_heads: int = 4
    num_basis: int = 5
    num_dims: int = 1

    @nn.compact
    def __call__(self, d: jax.Array, mask: Optional[jax.Array] = None):
        shape = (self.num_heads, self.num_basis, self.num_dims)
        a = self.param("a", init.constant(1), shape)
        b = self.param("b", init.constant(1), shape)
        if mask is None:
            mask = jnp.array([True])
        return rbf_network_bias(d, mask, a, b)


@jit
def rbf_network_bias(
    d: jax.Array,  # [B, Q, K, D] or [N, D]
    mask: jax.Array,  # [B, Q, K, D] or [N, D]
    a: jax.Array,  # [H, F, D]
    b: jax.Array,  # [H, F, D]
):
    """Returns an attention bias matrix of shape `[B, H, Q, K]`.

    For each dimension `D` of `d`, this maps a set of parameters
    `[H, F]` over each value in `[B, Q, K]`, yielding a matrix
    of shape `[B, Q, K, H, F, D]`. These values are then summed
    by attention head and reordered to yield `[B, H, Q, K]`.
    """
    if d.ndim == 2:  # GNN edges to attention map format
        d, mask = d[:, None, None, :], mask[:, None, None, :]
    mask = mask[..., None, None, None]  # [B, Q, K, 1, 1, 1]
    d = d[..., None, None, :]  # [B, Q, K, 1, 1, D]
    a = a[None, None, None, :, :, :]  # [1, 1, 1, H, F, D]
    b = b[None, None, None, :, :, :]  # [1, 1, 1, H, F, D]
    # double `jnp.where` to avoid NaN gradients: http://bit.ly/4aNgBjw
    d_m = jnp.where(mask, d, 0)
    d_rbf = a * jnp.exp(-b * d_m**2)  # [B, Q, K, H, F, D]
    d_rbf = jnp.where(mask, d_rbf, -jnp.inf)
    d_rbf = d_rbf.sum(axis=(-2, -1))  # [B, Q, K, H]
    return d_rbf.transpose(0, 3, 1, 2)  # [B, H, Q, K]


@jit
def scanned_rbf_network_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F, D]
    b: jax.Array,  # [H, F, D]
):
    d = vmap(l2_dist)(qs_meta, ks_meta)[..., None]  # [B, Q, K, D=1]
    mask = jnp.isfinite(d)
    return rbf_network_bias(d, mask, a, b)


class TISABias(nn.Module):
    """[Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""

    num_basis: int = 5
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array, mask: Optional[jax.Array] = None):
        a = self.param("a", init.constant(1), (self.num_heads, self.num_basis))
        b = self.param("b", init.constant(1), (self.num_heads, self.num_basis))
        c = self.param("c", init.constant(1), (self.num_heads, self.num_basis))
        if mask is None:
            mask = jnp.array([True])
        return tisa_bias(d, mask, a, b, c)


@jit
def tisa_bias(
    d: jax.Array,  # [B, Q, K, D] or [N, D]
    mask: jax.Array,  # [B, Q, K] or [N]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
):
    (B, Q, K), (H, F) = d.shape, a.shape
    a, b, c = a.flatten(), b.flatten(), c.flatten()
    x = vmap(tisa_rbf_basis, in_axes=(None, None, 0, 0, 0), out_axes=1)(
        d, mask, a, b, c
    )
    return x.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H*F, Q, K] -> [B, H, Q, K]


@jit
def tisa_rbf_basis(
    d: jax.Array,
    mask: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
):
    """Equation 5 in [Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""
    # double `jnp.where` to avoid NaN gradients: http://bit.ly/4aNgBjw
    d_m = jnp.where(mask, d, 0)
    d_m_tisa = a * jnp.exp(-jnp.abs(b) * (d_m - c) ** 2)
    return jnp.where(mask, d_m_tisa, -jnp.inf)


@jit
def scanned_tisa_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
):
    d = vmap(l2_dist)(qs_meta, ks_meta)  # [B, Q, K]
    mask = jnp.isfinite(d)
    return tisa_bias(d[..., None], mask, a, b, c)


def zero_bias(qs_meta, ks_meta):
    (B, Q, _M), K = qs_meta.shape, ks_meta.shape[1]
    return jnp.zeros((B, 1, Q, K))
