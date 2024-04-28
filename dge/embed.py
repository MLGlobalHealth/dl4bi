from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import Partial

from .mlp import MLP


class LearnableEmbedding(nn.Module):
    embed_s: Callable = lambda s: s
    post_process: Callable = MLP([256, 256])

    @nn.compact
    def __call__(self, s: jax.Array, training=False):
        return self.post_process(self.embed_s(s), training)


class FixedSinusoidalEmbedding(nn.Module):
    r"""Fixed sinusoidal positional encoding from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    $$
    \begin{aligned}
        \text{pe}(s,2i)&=\frac{s}{10000^{2i/d}} \\\\
        \text{pe}(s,2i+1)&=\frac{s}{10000^{2i/d}}
    \end{aligned}
    $$

    .. warning:: This maps each element of the last dimension independently and
        then concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times DE}$.
        For example, a 3 dimensional point `(x,y,z)` will get mapped to a `3*embed_dim`
        embedding.
    """

    embed_dim: int = 256

    @nn.compact
    def __call__(self, s: jax.Array):
        B, L, D = s.shape
        return _pe_attn_sinusoidal(self.embed_dim)(s).reshape(B, L, D * self.embed_dim)


def _pe_attn_sinusoidal(d: int):
    f = lambda i, s: s / (10000 ** (2 * i / d))
    return _pe_sinusoidal(f, d)


# TODO(danj): period makes a huge difference here...
def _pe_nerf_sinusoidal(d: int):
    f = lambda i, s: 2**i * jnp.pi * s
    return _pe_sinusoidal(f, d)


def _pe_sinusoidal(f: Callable, d: int):
    i = jnp.arange(d // 2)
    vf = lambda s: jnp.apply_along_axis(vmap(Partial(vmap(f, (0, None)), i)), -1, s)
    return jit(lambda s: jnp.concatenate([jnp.sin(vf(s)), jnp.cos(vf(s))], -1))


class NeRFEmbedding(nn.Module):
    r"""Positional encoding with MLP from ["NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"](https://arxiv.org/abs/2003.08934).

    $$
    \begin{aligned}
        F_\Theta&=F_\Theta^\prime\circ\gamma \\\\
        \gamma(s)&=[\sin(2^0\pi s), \cos(2^0\pi s),\ldots,\sin(2^{d-1}\pi s),\cos(2^{d-1}\pi s)] \\\\
    \end{aligned}
    $$

    .. warning:: This maps each element of the last dimension independently and
        then concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times DE}$.
        For example, a 3 dimensional point `(x,y,z)` will get mapped to a `3*embed_dim`
        embedding.
    """

    embed_dim: int = 256

    @nn.compact
    def __call__(self, s: jax.Array):
        B, L, D = s.shape
        return _pe_nerf_sinusoidal(self.embed_dim)(s).reshape(B, L, D * self.embed_dim)


class GaussianFourierEmbedding(nn.Module):
    r"""Gaussian Fourier Feature (GFF) positional encoding from ["Fourier Features Let Networks Learn
        High Frequency Functions in Low Dimensional Domains"](https://arxiv.org/abs/2006.10739).

    Must provide a starting $\mathbf{B}$, which can be generated with `B =
    random.normal(key, (embed_dim // 2, input_dim))`.

    $$
    \begin{aligned}
        \gamma(\mathbf{v})&=[\cos (2 \pi \mathbf{B v}), \sin (2 \pi \mathbf{B} \mathbf{v})] \\\\
        \mathbf{B}&\sim\mathcal{N}(0, \sigma^2)
    \end{aligned}
    $$

    .. warning:: This maps every element of the last dimension together:
        $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times E}$.
    """

    B: jax.Array  # [embed_dim, input_dim]
    var_init: Callable = nn.initializers.constant(10.0)

    @nn.compact
    def __call__(self, s):
        embed_dim, input_dim = self.B.shape
        var = self.param("var", self.var_init, (1,))
        return _pe_gaussian_fourier(self.B, var)(s)


def _pe_gaussian_fourier(B, var):
    s_proj = lambda s: (2.0 * jnp.pi * s) @ (var * B).T
    return jit(
        lambda s: jnp.concatenate([jnp.sin(s_proj(s)), jnp.cos(s_proj(s))], axis=-1)
    )
