"""Embedding layers and positional encoding helpers."""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.nn.initializers as init
import jax.numpy as jnp
from einops import repeat
from jax import jit, random, vmap
from jax.tree_util import Partial


class IDEmbedding(nn.Module):
    """Embeds IDs, which are assumed to be the first channel of the last dimension of `x`."""

    num_ids: int
    num_features: int
    channel: int = 0

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False):
        """Replace the configured ID channel with a learned embedding."""
        ids = nn.Embed(self.num_ids, self.num_features)(
            x[..., self.channel].astype(jnp.int32)
        )
        x_before, x_after = x[..., : self.channel], x[..., self.channel + 1 :]
        return jnp.concat([x_before, ids, x_after], axis=-1)


class ResidualEmbedding(nn.Module):
    """Returns [x, embed(x)]."""

    embed: Callable

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False):
        """Concatenate the input with the output of the wrapped embedding."""
        return jnp.concatenate([x, self.embed(x, training)], axis=-1)


class FixedSinusoidalEmbedding(nn.Module):
    r"""Fixed sinusoidal positional encoding from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

    $$
    \begin{aligned}
        \text{pe}(s,2i)&=\frac{s}{\text{max-len}^{2i/d}} \\\\
        \text{pe}(s,2i+1)&=\frac{s}{\text{max-len}^{2i/d}}
    \end{aligned}
    $$

    .. warning::
        This maps each element of the last dimension independently and then
        concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times
        DE}$. For example, a 3 dimensional point `(x,y,z)` will get mapped to a
        `3*embed_dim` embedding.
    """

    embed_dim: int = 256
    max_len: int = 10000

    @nn.compact
    def __call__(self, s: jax.Array, training: bool = False):
        """Encode coordinates with fixed sinusoidal positional features."""
        B, L, D = s.shape
        return _pe_attn_sinusoidal(
            self.embed_dim,
            self.max_len,
        )(s).reshape(B, L, D * self.embed_dim)


def _pe_attn_sinusoidal(d: int, max_len: float = 10000):
    """Build the fixed sinusoidal encoding used by Transformers."""
    f = lambda i, s: s / (max_len ** (2 * i / d))
    return _pe_sinusoidal(f, d)


# TODO(danj): period makes a huge difference here...
def _pe_nerf_sinusoidal(d: int):
    """Build the NeRF sinusoidal encoding basis."""
    f = lambda i, s: 2**i * jnp.pi * s
    return _pe_sinusoidal(f, d)


def _pe_sinusoidal(f: Callable, d: int):
    """Create a sinusoidal positional encoding from a frequency function."""
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

    .. warning::
        This maps each element of the last dimension independently and then
        concatenates: $\mathbb{R}^{\ldots\times D}\to\mathbb{R}^{\ldots\times
        DE}$. For example, a 3 dimensional point `(x,y,z)` will get mapped to a
        `3*embed_dim` embedding.
    """

    embed_dim: int = 256

    @nn.compact
    def __call__(self, s: jax.Array, training: bool = False):
        """Encode coordinates with NeRF-style sinusoidal features."""
        B, L, D = s.shape
        return _pe_nerf_sinusoidal(self.embed_dim)(s).reshape(B, L, D * self.embed_dim)


class GaussianFourierEmbedding(nn.Module):
    r"""Gaussian Fourier Feature (GFF) positional encoding from ["Fourier Features Let Networks Learn
        High Frequency Functions in Low Dimensional Domains"](https://arxiv.org/abs/2006.10739).

    $$
    \begin{aligned}
        \gamma(\mathbf{v})&=[\cos (2 \pi \mathbf{B v}), \sin (2 \pi \mathbf{B} \mathbf{v})] \\\\
        \mathbf{B}&\sim\mathcal{N}(0, \sigma^2)
    \end{aligned}
    $$

    .. warning::
        This maps every element of the last dimension together and produces
        both a sine and cosine feature: $\mathbb{R}^{\ldots\times D}
        \to\mathbb{R}^{\ldots\times E}$.
    """

    embed_dim: int = 256
    std: float = 4.0

    @nn.compact
    def __call__(self, s, training: bool = False):
        """Encode coordinates with Gaussian Fourier features."""
        gen_B = lambda rng: random.normal(rng, (self.embed_dim // 2, s.shape[-1]))
        B = self.variable("projections", "B", lambda: gen_B(self.make_rng("params")))
        return _pe_gaussian_fourier(B.value, self.std)(s)


def _pe_gaussian_fourier(B: jax.Array, std: float):
    """Build a Gaussian Fourier feature projection with fixed frequencies."""
    s_proj = lambda s: (2.0 * jnp.pi * s) @ (std * B).T
    return jit(
        lambda s: jnp.concatenate([jnp.sin(s_proj(s)), jnp.cos(s_proj(s))], axis=-1)
    )


class RBFRandomFourierFeatures(nn.Module):
    r"""Generate Random Fourier Features with learnable lengthscale and weight.

    $\text{RFF}(x)\approx a * \text{RBF}(d^2 / ls^2)$

    Args:
        embed_dim: Embedding dimension.

    Returns:
        An instance of `RBFRandomFourierFeatures` embedding function.
    """

    embed_dim: int = 256
    num_heads: int = 4

    @nn.compact
    def __call__(self, s: jax.Array):
        """Generate random Fourier features for RBF-style similarity modeling."""
        H = self.num_heads
        a = self.param("a", init.constant(1.0), (1, 1, H, 1))
        ls = self.param("ls", init.constant(1.0), (1, 1, H, 1))
        omega = self.variable(
            "projections",
            "omega",
            lambda: _gen_omega(
                self.make_rng("params"),
                self.embed_dim,
                s.shape[-1],
            ),
        )
        b = self.variable(
            "projections",
            "b",
            lambda: random.uniform(
                self.make_rng("params"),
                shape=(self.embed_dim // 2,),
            ),
        )
        proj = jnp.einsum("B L S, S E -> B L E", s, omega.value)
        proj = repeat(proj, "B L E -> B L H E", H=H) / ls
        rffs = jnp.concatenate(
            [jnp.cos(proj + b.value), jnp.sin(proj + b.value)], axis=-1
        ) / jnp.sqrt(self.embed_dim // 2)
        return a * rffs


def _gen_omega(rng: jax.Array, embed_dim: int, s_dim: int):
    """Sample random Fourier frequencies for each input dimension."""
    omegas = []
    for _ in range(s_dim):
        rng_omega, rng = random.split(rng)
        omega = random.normal(rng_omega, shape=(embed_dim // 2,))
        omegas += [omega]
    return jnp.vstack(omegas)
