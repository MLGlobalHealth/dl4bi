from functools import partial
from typing import Callable, Optional

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax import jit, vmap
from sps.kernels import l2_dist

from .compose import ComposableModule


@partial(jit, static_argnames=("causal_t",))
def dist_spatiotemporal(
    q: jax.Array,  # [Q, D]
    r: jax.Array,  # [R, D]
    causal_t: bool = False,
):
    """Returns a spatiotemporal distance matrix of shape `[Q, R, 2]`.

    This assumes that the spatial dims of `q` and `r` are all  but the last,
    i.e. `{q,r}[:, :-1]`, and the temporal dim is the last dim,
    i.e. `{q,r}[:, -1]`.

    The entries in `[Q, R, 0]` represent the L2 spatial distance.
    The entries in `[Q, R, 1]` represent the signed L1 temporal distance.

    For example, for `q[i] = [x=2, y=3, t=1]` and `r[j] = [x=1, y=0, t=0]`,
    this will return `m[i, j] = [sqrt(10), -1]`.
    """
    d_s = dist_spatial(q[..., :-1], r[..., :-1])  # [Q, R] L2 dist
    d_t = -q[..., [-1]] + r[..., [-1]].T  # [Q, R] L1 temporal dist
    d = jnp.stack([d_s, d_t], axis=-1)  # [Q, R, 2]
    if causal_t:  # set distances to inf for future timesteps
        return jnp.where((d_t <= 0)[..., None], d, jnp.inf)
    return d


@jit
def dist_spatial(q: jax.Array, r: jax.Array):
    return l2_dist(q, r)[..., None]  # [Q, R, D=1]


class DistanceBias(nn.Module):
    num_heads: int = 4
    channel: int = 0

    @nn.compact
    def __call__(
        self,
        d: jax.Array,  # [B, Q, K, D] or [E, D]
        mask: Optional[jax.Array] = None,  # None or [B, Q, K] or [E]
    ):
        a = self.param("a", init.constant(-1), (self.num_heads,))
        if mask is None:
            mask = jnp.array([True])
        return distance_bias(d[..., self.channel], mask, a)


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


class RBFNetworkBias(ComposableModule):
    num_heads: int = 4
    num_basis: int = 5
    channel: int = 0

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
        return rbf_network_bias(d[..., self.channel], mask, a, b)


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


@partial(jit, static_argnames=("func", "channel"))
def scanned_rbf_network_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    func: Callable = dist_spatial,
    channel: int = 0,
):
    d = vmap(func)(qs_meta, ks_meta)[..., channel]  # [B, Q, K]
    mask = jnp.isfinite(d)
    return rbf_network_bias(d, mask, a, b)


class TISABias(ComposableModule):
    """[Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""

    num_heads: int = 4
    num_basis: int = 5
    channel: int = 0

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
        return tisa_bias(d[..., self.channel], mask, a, b, c)


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


@partial(jit, static_argnames=("func", "channel"))
def scanned_tisa_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
    func: Callable = dist_spatial,
    channel: int = 0,
):
    d = vmap(func)(qs_meta, ks_meta)[..., channel]  # [B, Q, K]
    mask = jnp.isfinite(d)
    return tisa_bias(d, mask, a, b, c)


@jit
def zero_bias(qs_meta, ks_meta):
    (B, Q, _M), K = qs_meta.shape, ks_meta.shape[1]
    return jnp.zeros((B, 1, Q, K))  # [B, H, Q, K]
