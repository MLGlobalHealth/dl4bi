import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.typing import ArrayLike


class DistanceBias(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array):
        d = jnp.repeat(d[:, None, ...], self.num_heads, axis=1)
        a = self.param("a", init.constant(-1), (1, self.num_heads, 1, 1))
        return a * d  # [B, H, Q, K]
    
class GraphDistanceBias(nn.Module):
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array, d2: jax.Array):
        d = jnp.repeat(d[:, None, ...], self.num_heads, axis=1)
        d2 = jnp.repeat(d2[:, None, ...], self.num_heads, axis=1)
        a = self.param("a", init.constant(-1), (1, self.num_heads, 1, 1))
        b = self.param("b", init.constant(-1), (1, self.num_heads, 1, 1))
        return a * d + b * d2 # [B, H, Q, K]


class TISABias(nn.Module):
    """[Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""

    num_basis: int = 5
    num_heads: int = 4

    @nn.compact
    def __call__(self, d: jax.Array):
        (B, Q, K), H, F = d.shape, self.num_heads, self.num_basis
        a = self.param("a", init.constant(1), (H * F,))
        b = self.param("b", init.constant(1), (H * F,))
        c = self.param("c", init.constant(0), (H * F,))
        # TODO(danj): convert vmap to scan
        x = vmap(rbf_basis, in_axes=(None, 0, 0, 0), out_axes=1)(d, a, b, c)
        return x.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H * F, Q, K] -> [B, H, Q, K]


# TODO(danj): remove absolute value on b to make this strictly more expressive?
@jit
def rbf_basis(d, a, b, c):
    """Equation 5 in [Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""
    return a * jnp.exp(-jnp.abs(b) * (d - c) ** 2)
