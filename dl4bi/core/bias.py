from typing import Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import l2_dist


class DistanceBias(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array):
        d = jnp.repeat(d[:, None, ...], self.num_heads, axis=1)
        a = self.param("a", init.constant(-1), (1, self.num_heads, 1, 1))
        return a * d  # [B, H, Q, K]


class RBFNetworkBias(nn.Module):
    num_basis: int = 5
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array, mask: Optional[jax.Array] = None):
        a = self.param("a", init.constant(1), (self.num_heads, self.num_basis))
        b = self.param("b", init.constant(1), (self.num_heads, self.num_basis))
        return rbf_network_bias(d, a, b, mask)


@jit
def rbf_network_bias(
    d: jax.Array,
    a: jax.Array,
    b: jax.Array,
    mask: Optional[jax.Array] = None,  # [B, Q, K]
):
    (B, Q, K), (H, F) = d.shape, a.shape
    a, b = a.flatten(), b.flatten()
    x = vmap(rbf_basis, in_axes=(None, 0, 0), out_axes=1)(d, a, b)
    x = x.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H * F, Q, K] -> [B, H, Q, K]
    if mask is not None:
        return jnp.where(mask[:, None, ...], x, -jnp.inf)
    return x


@jit
def rbf_basis(d, a, b):
    return a * jnp.exp(-b * d**2)


@jit
def scanned_rbf_network_bias(
    qs_meta: jax.Array,
    ks_meta: jax.Array,
    a: jax.Array,
    b: jax.Array,
):
    d = vmap(l2_dist)(qs_meta, ks_meta)  # [B, Q, K]
    mask = jnp.isfinite(d)
    return rbf_network_bias(d, a, b, mask)


class TISABias(nn.Module):
    """[Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""

    num_basis: int = 5
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array, mask: Optional[jax.Array] = None):
        a = self.param("a", init.constant(1), (self.num_heads, self.num_basis))
        b = self.param("b", init.constant(1), (self.num_heads, self.num_basis))
        c = self.param("c", init.constant(1), (self.num_heads, self.num_basis))
        return tisa_bias(d, a, b, c, mask)


@jit
def tisa_bias(
    d: jax.Array,  # [B, Q, K]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
    mask: Optional[jax.Array] = None,  # [B, Q, K]
):
    (B, Q, K), (H, F) = d.shape, a.shape
    a, b, c = a.flatten(), b.flatten(), c.flatten()
    x = vmap(tisa_rbf_basis, in_axes=(None, 0, 0, 0), out_axes=1)(d, a, b, c)
    x = x.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H*F, Q, K] -> [B, H, Q, K]
    if mask is not None:
        return jnp.where(mask[:, None, ...], x, -jnp.inf)
    return x


@jit
def tisa_rbf_basis(d, a, b, c):
    """Equation 5 in [Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""
    return a * jnp.exp(-jnp.abs(b) * (d - c) ** 2)


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
    return tisa_bias(d, a, b, c, mask)


def zero_bias(qs_meta, ks_meta):
    (B, Q, _M), K = qs_meta.shape, ks_meta.shape[1]
    return jnp.zeros((B, 1, Q, K))
