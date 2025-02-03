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
        x = vmap(tisa_rbf_basis, in_axes=(None, 0, 0, 0), out_axes=1)(d, a, b, c)
        return x.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H * F, Q, K] -> [B, H, Q, K]


@jit
def tisa_bias(
    qs_meta: jax.Array,  # [B, Q, M]
    ks_meta: jax.Array,  # [B, K, M]
    a: jax.Array,  # [H, F]
    b: jax.Array,  # [H, F]
    c: jax.Array,  # [H, F]
):
    (H, F), (B, Q), K = a.shape, qs_meta.shape[:2], ks_meta.shape[1]
    a, b, c = a.flatten(), b.flatten(), c.flatten()
    d = vmap(l2_dist)(qs_meta, ks_meta)  # [B, Q, K]
    bias = vmap(tisa_rbf_basis, in_axes=(None, 0, 0, 0), out_axes=1)(
        d, a, b, c
    )  # [B, H * F, Q, K]
    return bias.reshape(B, H, F, Q, K).sum(axis=2)  # [B, H, Q, K]


@jit
def tisa_rbf_basis(d, a, b, c):
    """Equation 5 in [Translation-Invariant Self-Attention (TISA)](https://arxiv.org/abs/2106.01950) Bias."""
    return a * jnp.exp(-jnp.abs(b) * (d - c) ** 2)


def zero_bias(qs_x, ks_x):
    (B, Q, _X), K = qs_x.shape, ks_x.shape[1]
    return jnp.zeros((B, 1, Q, K))
