from typing import Callable

import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from flax import linen as nn


class HyperLoRAqkv(nn.Module):
    rank: int = 16
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = init.orthogonal()

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array):
        (B, L, D), D_z, R = x.shape, z.shape[-1], self.rank
        # shared base projection
        qkv = nn.Dense(3 * D, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # shared down projection for LoRA update
        V = self.param("V", init.lecun_normal(), (R, D), self.dtype)
        xV = jnp.einsum("B L D, R D -> B L R", x, V)
        # independent scales and up projections for LoRA update
        scale = nn.Dense(3 * R, dtype=self.dtype, kernel_init=self.kernel_init)
        if z.ndim == 3:  # per token condition
            h = scale(z.reshape(B * L, D_z)).reshape(B, L, 3, R)
            s_q, s_k, s_v = jnp.split(h, 3, axis=2)  # [B, L, 1, R]
            s_q, s_k, s_v = map(lambda x: x[..., 0, :], (s_q, s_k, s_v))
        else:  # per batch element condition
            h = scale(z).reshape(B, 3, R)
            s_q, s_k, s_v = jnp.split(h, 3, axis=1)  # [B, 1, R]
        U_q = self.param("U_q", init.zeros, (D, R), self.dtype)
        U_k = self.param("U_k", init.zeros, (D, R), self.dtype)
        U_v = self.param("U_v", init.zeros, (D, R), self.dtype)
        delta_q = jnp.einsum("B L R, R D -> B L D", xV * s_q, U_q.T)
        delta_k = jnp.einsum("B L R, R D -> B L D", xV * s_k, U_k.T)
        delta_v = jnp.einsum("B L R, R D -> B L D", xV * s_v, U_v.T)
        return q + delta_q, k + delta_k, v + delta_v


class HyperLoRA(nn.Module):
    """
    Generic Dense(x) + conditional low-rank ΔW(z) using LoRA factorization.

    y = base(x) + U @ ( (V @ x) ⊙ s(z) )
    """

    out_dim: int = 128
    rank: int = 16
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = init.orthogonal()

    @nn.compact
    def __call__(self, x: jax.Array, z: jax.Array):
        (B, L, D_in), D_out, D_z, R = x.shape, self.out_dim, z.shape[-1], self.rank
        V = self.param("V", init.lecun_normal(), (R, D_in), self.dtype)
        U = self.param("U", init.zeros, (D_out, R), self.dtype)
        scale = nn.Dense(R, dtype=self.dtype, kernel_init=self.kernel_init)
        h = nn.Dense(D_out, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        xV = jnp.einsum("B L D, R D -> B L R", x, V)
        if z.ndim == 3:  # per token condition
            s = scale(z.reshape(B * L, D_z)).reshape(B, L, R)
        else:  # per batch element condition
            s = scale(z).reshape(B, 1, R)
        delta = jnp.einsum("B L R, R D -> B L D", xV * s, U.T)
        return h + delta
