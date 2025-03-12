from functools import partial
from typing import Callable, Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax import jit, vmap

from .sim import dist_spatial


class DistanceBias(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        d: jax.Array,  # [B, Q, K] or [E, D]
        mask: Optional[jax.Array] = None,  # None or [B, Q, K] or [E]
    ):
        a = self.param("a", init.constant(-1), (self.num_heads,))
        if mask is None:
            mask = jnp.array([True])
        return distance_bias(d, mask, a)  # [B, H, Q, K]


@jit
def distance_bias(
    d: jax.Array,  # [B, Q, K] or [E]
    mask: jax.Array,  # [B, Q, K] or [E]
    a: jax.Array,  # [H]
):
    """Returns an attention bias matrix of shape `[B, H, Q, K]`."""
    is_edges = d.ndim == 1
    if is_edges:  # GNN edges to attention map format
        d, mask = d[:, None, None], mask[:, None, None]  # [B, Q=1, K=1]
    d, mask = d[:, None, ...], mask[:, None, ...]  # [B, 1, Q, K]
    a = a[None, :, None, None]  # [1, H, 1, 1]
    # double `jnp.where` to avoid NaN gradients: http://bit.ly/4aNgBjw
    d_m = jnp.where(mask, d, 0)
    d_a = jnp.where(mask, a * d_m, -jnp.inf)  # [B, H, Q, K]
    if is_edges:
        return d_a.squeeze()  # [B=E, H, Q=1, K=1] -> [E, H]
    return d_a  # [B, H, Q, K]


class RBFNetworkBias(nn.Module):
    num_heads: int = 4
    num_basis: int = 5

    @nn.compact
    def __call__(
        self,
        d: jax.Array,  # [B, Q, K, D] or [E, D]
        mask: Optional[jax.Array] = None,  # None or [B, Q, K] or [E]
    ):
        a = self.param("a", init.constant(1), (self.num_heads, self.num_basis))
        b = self.param("b", init.constant(1), (self.num_heads, self.num_basis))
        if mask is None:
            mask = jnp.array([True])
        return rbf_network_bias(d, mask, a, b)  # [B, H, Q, K]


@jit
def rbf_network_bias(
    d: jax.Array,  # [B, Q, K] or [E]
    mask: jax.Array,  # [B, Q, K] or [E]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
):
    """Returns an attention bias matrix of shape `[B, H, Q, K]`."""
    is_edges = d.ndim == 1
    if is_edges:  # GNN edges to attention map format
        d, mask = d[:, None, None], mask[:, None, None]  # [B, Q=1, K=1]
    mask = mask[:, None, :, :, None]  # [B, 1, Q, K, 1]
    d = d[:, None, :, :, None]  # [B, 1, Q, K, 1]
    a = a[None, :, None, None, :]  # [1, H, 1, 1, F]
    b = b[None, :, None, None, :]  # [1, H, 1, 1, F]
    # double `jnp.where` to avoid NaN gradients: http://bit.ly/4aNgBjw
    d_m = jnp.where(mask, d, 0)
    d_rbf = a * jnp.exp(-b * d_m**2)  # [B, H, Q, K, F]
    d_rbf = jnp.where(mask, d_rbf, -jnp.inf)
    d_rbf = d_rbf.sum(axis=-1)  # [B, H, Q, K]
    if is_edges:
        return d_rbf.squeeze()  # [B=E, H, Q=1, K=1] -> [E, H]
    return d_rbf  # [B, H, Q, K]


@partial(jit, static_argnames=("func",))
def scanned_rbf_network_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    func: Callable = dist_spatial,
):
    d = vmap(func)(qs_meta, ks_meta)  # [B, Q, K]
    mask = jnp.isfinite(d)
    return rbf_network_bias(d, mask, a, b)


class TISABias(nn.Module):
    """[Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""

    num_heads: int = 4
    num_basis: int = 5

    @nn.compact
    def __call__(
        self,
        d: jax.Array,  # [B, Q, K, D] or [E, D]
        mask: Optional[jax.Array] = None,  # None or [B, Q, K] or [E]
    ):
        a = self.param("a", init.constant(1), (self.num_heads, self.num_basis))
        b = self.param("b", init.constant(1), (self.num_heads, self.num_basis))
        c = self.param("c", init.constant(1), (self.num_heads, self.num_basis))
        if mask is None:
            mask = jnp.array([True])
        return tisa_bias(d, mask, a, b, c)  # [B, H, Q, K]


@jit
def tisa_bias(
    d: jax.Array,  # [B, Q, K] or [E]
    mask: jax.Array,  # [B, Q, K] or [E]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
):
    """Equation 5 in [Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""
    is_edges = d.ndim == 1
    if is_edges:  # GNN edges to attention map format
        d, mask = d[:, None, None], mask[:, None, None]  # [B, Q=1, K=1]
    mask = mask[:, None, :, :, None]  # [B, 1, Q, K, 1]
    d = d[:, None, :, :, None]  # [B, 1, Q, K, 1]
    a = a[None, :, None, None, :]  # [1, H, 1, 1, F]
    b = b[None, :, None, None, :]  # [1, H, 1, 1, F]
    c = c[None, :, None, None, :]  # [1, H, 1, 1, F]
    # double `jnp.where` to avoid NaN gradients: http://bit.ly/4aNgBjw
    d_m = jnp.where(mask, d, 0)
    d_tisa = a * jnp.exp(-b * (d_m - c) ** 2)  # [B, H, Q, K, F]
    d_tisa = jnp.where(mask, d_tisa, -jnp.inf)
    d_tisa = d_tisa.sum(axis=-1)  # [B, H, Q, K]
    if is_edges:
        return d_tisa.squeeze()  # [B=E, H, Q=1, K=1] -> [E, H]
    return d_tisa  # [B, H, Q, K]


@partial(jit, static_argnames=("func",))
def scanned_tisa_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
    func: Callable = dist_spatial,
):
    d = vmap(func)(qs_meta, ks_meta)  # [B, Q, K]
    mask = jnp.isfinite(d)
    return tisa_bias(d, mask, a, b, c)


@jit
def zero_bias(qs_meta, ks_meta):
    (B, Q, _M), K = qs_meta.shape, ks_meta.shape[1]
    return jnp.zeros((B, 1, Q, K))  # [B, H, Q, K]
